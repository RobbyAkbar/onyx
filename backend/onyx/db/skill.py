"""DB operations for skill rows.

Access model:
- `VIEW` is the skills UI/read API policy. It excludes external-app-backed rows,
  applies user visibility, and lets admins view all non-external-app rows.
- `EDIT` is the custom-skill mutation policy. It excludes external-app-backed
  and built-in rows, and only returns rows the user can modify.
- `USE` is the runtime/sandbox policy. It applies user visibility without an
  admin bypass, requires enabled rows, includes authenticated external-app-backed
  rows, and hides unavailable built-ins.

Delete is a hard delete — `delete_skill` removes the row and returns its
`bundle_file_id` so the caller can drop the blob from the file store
immediately (skills sync via S3-backed bundles, so blob retention isn't
needed).

These helpers never commit — callers control the transaction boundary so a
multi-step admin flow (e.g. create row + replace shares) can roll back atomically.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from enum import Enum
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy import ColumnElement
from sqlalchemy import delete
from sqlalchemy import exists
from sqlalchemy import or_
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.db.enums import SandboxStatus
from onyx.db.enums import SkillSharePermission
from onyx.db.external_app import is_user_authenticated_for_app
from onyx.db.models import ExternalApp
from onyx.db.models import ExternalAppUserCredential
from onyx.db.models import Sandbox
from onyx.db.models import Skill
from onyx.db.models import Skill__User
from onyx.db.models import Skill__UserGroup
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.utils import is_fk_violation
from onyx.db.utils import is_unique_violation
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.skills.built_in import BUILT_IN_SKILLS

SKILL_SLUG_UNIQUE_CONSTRAINT = "uq_skill_slug"


class SkillAccessPolicy(str, Enum):
    VIEW = "view"
    EDIT = "edit"
    USE = "use"


def _is_skill_author_clause(user: User) -> ColumnElement[bool]:
    return and_(
        Skill.author_user_id == user.id,
        Skill.built_in_skill_id.is_(None),
    )


def _user_share_exists(
    user: User,
    permission: SkillSharePermission | None = None,
) -> ColumnElement[bool]:
    stmt = (
        select(Skill__User.skill_id)
        .where(Skill__User.skill_id == Skill.id)
        .where(Skill__User.user_id == user.id)
    )
    if permission is not None:
        stmt = stmt.where(Skill__User.permission == permission)
    return stmt.exists()


def _group_share_exists(
    user: User,
    permission: SkillSharePermission | None = None,
) -> ColumnElement[bool]:
    stmt = (
        select(Skill__UserGroup.skill_id)
        .join(
            User__UserGroup,
            User__UserGroup.user_group_id == Skill__UserGroup.user_group_id,
        )
        .where(Skill__UserGroup.skill_id == Skill.id)
        .where(User__UserGroup.user_id == user.id)
    )
    if permission is not None:
        stmt = stmt.where(Skill__UserGroup.permission == permission)
    return stmt.exists()


def _any_share_exists() -> ColumnElement[bool]:
    user_share_exists = (
        select(Skill__User.skill_id).where(Skill__User.skill_id == Skill.id).exists()
    )
    group_share_exists = (
        select(Skill__UserGroup.skill_id)
        .where(Skill__UserGroup.skill_id == Skill.id)
        .exists()
    )
    return or_(user_share_exists, group_share_exists)


def _curator_group_management_clause(user: User) -> ColumnElement[bool]:
    scoped_group_ids = select(User__UserGroup.user_group_id).where(
        User__UserGroup.user_id == user.id
    )
    scoped_group_share_exists = (
        select(Skill__UserGroup.skill_id)
        .join(
            User__UserGroup,
            User__UserGroup.user_group_id == Skill__UserGroup.user_group_id,
        )
        .where(Skill__UserGroup.skill_id == Skill.id)
        .where(User__UserGroup.user_id == user.id)
    )

    if user.role == UserRole.CURATOR:
        scoped_group_ids = scoped_group_ids.where(User__UserGroup.is_curator.is_(True))
        scoped_group_share_exists = scoped_group_share_exists.where(
            User__UserGroup.is_curator.is_(True)
        )

    no_group_share_outside_scope = ~exists().where(
        Skill__UserGroup.skill_id == Skill.id
    ).where(Skill__UserGroup.user_group_id.notin_(scoped_group_ids)).correlate(Skill)
    return and_(scoped_group_share_exists.exists(), no_group_share_outside_scope)


def _user_view_clause(user: User, *, admin_bypass: bool) -> ColumnElement[bool]:
    if admin_bypass and user.role == UserRole.ADMIN:
        return true()

    return or_(
        Skill.is_public.is_(True),
        _user_share_exists(user),
        _group_share_exists(user),
        _is_skill_author_clause(user),
    )


def _user_edit_clause(user: User) -> ColumnElement[bool]:
    editor_share_clause = or_(
        _user_share_exists(user, SkillSharePermission.EDITOR),
        _group_share_exists(user, SkillSharePermission.EDITOR),
        and_(
            Skill.is_public.is_(True),
            Skill.public_permission == SkillSharePermission.EDITOR,
        ),
    )

    if user.role == UserRole.ADMIN:
        return or_(
            _is_skill_author_clause(user),
            editor_share_clause,
            Skill.is_public.is_(True),
            _any_share_exists(),
        )

    if user.role in (UserRole.CURATOR, UserRole.GLOBAL_CURATOR):
        return or_(
            _is_skill_author_clause(user),
            editor_share_clause,
            _curator_group_management_clause(user),
        )

    return or_(
        _is_skill_author_clause(user),
        editor_share_clause,
    )


def _exclude_unavailable_built_ins(
    stmt: Select[tuple[Skill]], db_session: Session
) -> Select[tuple[Skill]]:
    """Hide built-ins whose codified ``is_available(db)`` returns False.
    User-facing reads use this; admin VIEW reads don't (admins see all rows)."""
    unavailable = [
        d.built_in_skill_id
        for d in BUILT_IN_SKILLS.values()
        if not d.is_available(db_session)
    ]
    if not unavailable:
        return stmt
    return stmt.where(
        or_(
            Skill.built_in_skill_id.is_(None),
            Skill.built_in_skill_id.notin_(unavailable),
        )
    )


def _skill_ids_blocked_by_external_app_auth(
    user: User, db_session: Session
) -> list[UUID]:
    """Skill ids to withhold from *user*'s sandbox: external-app-backed
    skills the user has not authenticated for.

    Each external app is left-joined to this user's credential row; an app
    the user can't use yet (missing required credential keys) has its skill
    blocked. Apps that need no per-user credentials, or that the user has
    already configured, are not blocked.
    """
    rows = db_session.execute(
        select(ExternalApp, ExternalAppUserCredential).join(
            ExternalAppUserCredential,
            and_(
                ExternalAppUserCredential.external_app_id == ExternalApp.id,
                ExternalAppUserCredential.user_id == user.id,
            ),
            isouter=True,
        )
    ).all()
    return [
        app.skill_id
        for app, user_cred in rows
        if not is_user_authenticated_for_app(app, user_cred)
    ]


def _skill_select(*, order_by_name: bool) -> Select[tuple[Skill]]:
    stmt = select(Skill).options(
        selectinload(Skill.author),
        selectinload(Skill.user_shares).selectinload(Skill__User.user),
        selectinload(Skill.group_shares).selectinload(Skill__UserGroup.user_group),
    )
    if order_by_name:
        stmt = stmt.order_by(Skill.name)
    return stmt


def _skill_select_for_access_policy(
    *,
    policy: SkillAccessPolicy,
    db_session: Session,
    user: User,
    order_by_name: bool,
) -> Select[tuple[Skill]]:
    stmt = _skill_select(order_by_name=order_by_name)

    match policy:
        case SkillAccessPolicy.VIEW:
            stmt = stmt.where(Skill.id.notin_(select(ExternalApp.skill_id)))
            stmt = stmt.where(_user_view_clause(user, admin_bypass=True))
            if user.role == UserRole.ADMIN:
                return stmt
            stmt = stmt.where(or_(Skill.enabled.is_(True), _user_edit_clause(user)))
            return _exclude_unavailable_built_ins(stmt, db_session)

        case SkillAccessPolicy.EDIT:
            return (
                stmt.where(Skill.id.notin_(select(ExternalApp.skill_id)))
                .where(Skill.built_in_skill_id.is_(None))
                .where(_user_edit_clause(user))
            )

        case SkillAccessPolicy.USE:
            blocked_skill_ids = _skill_ids_blocked_by_external_app_auth(
                user, db_session
            )
            stmt = stmt.where(Skill.enabled.is_(True)).where(
                Skill.id.notin_(blocked_skill_ids)
            )
            stmt = stmt.where(_user_view_clause(user, admin_bypass=False))
            return _exclude_unavailable_built_ins(stmt, db_session)

    raise ValueError(f"Unknown skill access policy: {policy}")


def visible_skill_ids_for_user(user: User, db_session: Session) -> set[UUID]:
    """Enabled skill ids the user can see, including external-app-backed rows.

    Used by the external-app API to decide which apps a user may connect. This
    deliberately does not apply the per-user external-app credential gate.
    """
    stmt = (
        select(Skill.id)
        .where(Skill.enabled.is_(True))
        .where(_user_view_clause(user, admin_bypass=False))
    )
    return set(db_session.scalars(stmt))


def affected_user_ids_for_skill(skill: Skill, db_session: Session) -> set[UUID]:
    """Return user IDs with a running sandbox that should contain this skill.

    Deliberately does not filter by ``enabled``: disable/delete flows still need
    the previous recipients so the push pipeline can remove the skill files.
    """
    if skill.is_public:
        stmt = select(Sandbox.user_id).where(Sandbox.status == SandboxStatus.RUNNING)
        return set(db_session.scalars(stmt))

    group_share_stmt = (
        select(Sandbox.user_id)
        .join(
            User__UserGroup,
            User__UserGroup.user_id == Sandbox.user_id,
        )
        .join(
            Skill__UserGroup,
            Skill__UserGroup.user_group_id == User__UserGroup.user_group_id,
        )
        .where(Skill__UserGroup.skill_id == skill.id)
        .where(Sandbox.status == SandboxStatus.RUNNING)
    )
    user_ids = set(db_session.scalars(group_share_stmt))

    user_share_stmt = (
        select(Sandbox.user_id)
        .join(
            Skill__User,
            Skill__User.user_id == Sandbox.user_id,
        )
        .where(Skill__User.skill_id == skill.id)
        .where(Sandbox.status == SandboxStatus.RUNNING)
    )
    user_ids |= set(db_session.scalars(user_share_stmt))

    if skill.author_user_id is not None:
        author_stmt = (
            select(Sandbox.user_id)
            .where(Sandbox.user_id == skill.author_user_id)
            .where(Sandbox.status == SandboxStatus.RUNNING)
        )
        user_ids |= set(db_session.scalars(author_stmt))

    return user_ids


def list_skills(
    *,
    policy: SkillAccessPolicy,
    db_session: Session,
    user: User,
) -> Sequence[Skill]:
    stmt = _skill_select_for_access_policy(
        policy=policy,
        db_session=db_session,
        user=user,
        order_by_name=True,
    )
    return list(db_session.scalars(stmt))


def fetch_skill(
    skill_id: UUID,
    *,
    policy: SkillAccessPolicy,
    db_session: Session,
    user: User,
) -> Skill | None:
    stmt = _skill_select_for_access_policy(
        policy=policy,
        db_session=db_session,
        user=user,
        order_by_name=False,
    ).where(Skill.id == skill_id)
    return db_session.scalars(stmt).one_or_none()


def fetch_skill_by_id_for_system(skill_id: UUID, db_session: Session) -> Skill | None:
    """Fetch a skill by id without applying user access policy.

    Only use this for system flows that have already made an authorization
    decision, such as post-commit reloads or sandbox invalidation.
    """
    stmt = _skill_select(order_by_name=False).where(Skill.id == skill_id)
    return db_session.scalars(stmt).one_or_none()


def _add_skill_with_unique_slug__no_commit(
    skill: Skill,
    *,
    slug: str,
    db_session: Session,
) -> Skill:
    existing = db_session.scalars(select(Skill.id).where(Skill.slug == slug)).first()
    if existing is not None:
        raise OnyxError(
            OnyxErrorCode.DUPLICATE_RESOURCE,
            f"A skill with slug '{slug}' already exists.",
        )

    db_session.add(skill)
    try:
        db_session.flush()
    except IntegrityError as e:
        if is_unique_violation(e, SKILL_SLUG_UNIQUE_CONSTRAINT):
            raise OnyxError(
                OnyxErrorCode.DUPLICATE_RESOURCE,
                f"A skill with slug '{slug}' already exists.",
            ) from e
        raise
    return skill


def create_skill__no_commit(
    *,
    slug: str,
    name: str,
    description: str,
    bundle_file_id: str,
    bundle_sha256: str,
    is_public: bool,
    public_permission: SkillSharePermission = SkillSharePermission.VIEWER,
    author_user_id: UUID | None,
    db_session: Session,
) -> Skill:
    skill = Skill(
        slug=slug,
        name=name,
        description=description,
        bundle_file_id=bundle_file_id,
        bundle_sha256=bundle_sha256,
        is_public=is_public,
        public_permission=public_permission,
        author_user_id=author_user_id,
        enabled=True,
    )
    return _add_skill_with_unique_slug__no_commit(
        skill,
        slug=slug,
        db_session=db_session,
    )


def create_built_in_skill_row__no_commit(
    *,
    built_in_skill_id: str,
    name: str,
    description: str,
    is_public: bool,
    enabled: bool,
    author_user_id: UUID | None = None,
    public_permission: SkillSharePermission = SkillSharePermission.VIEWER,
    db_session: Session,
) -> Skill:
    """Create a built-in-style ``Skill`` row: ``built_in_skill_id`` set,
    ``slug == built_in_skill_id`` (the stable on-disk dir name), bundle fields
    NULL (per the XOR check constraint). Used for external-app providers, whose
    rows are created on demand rather than seeded.

    Because the slug is the (globally unique) built-in id, a tenant can hold at
    most one row per provider — a second attempt raises
    ``OnyxError(DUPLICATE_RESOURCE)``, which is the desired "connect Slack once"
    behaviour.
    """
    skill = Skill(
        slug=built_in_skill_id,
        name=name,
        description=description,
        built_in_skill_id=built_in_skill_id,
        bundle_file_id=None,
        bundle_sha256=None,
        is_public=is_public,
        public_permission=public_permission,
        author_user_id=author_user_id,
        enabled=enabled,
    )
    return _add_skill_with_unique_slug__no_commit(
        skill,
        slug=built_in_skill_id,
        db_session=db_session,
    )


def replace_skill_bundle(
    *,
    skill: Skill,
    new_bundle_file_id: str,
    new_bundle_sha256: str,
    new_name: str,
    new_description: str,
    db_session: Session,
) -> str:
    """Swap a custom skill's bundle blob and refresh its display metadata.

    Returns the old bundle file id so the caller can delete the old blob from
    FileStore after the transaction commits.

    Name and description come from the new bundle's SKILL.md frontmatter so
    the DB row stays in lockstep with what's actually pushed to sandboxes.

    Rejects built-in rows — they have no bundle.
    """
    if skill.built_in_skill_id is not None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Skill '{skill.slug}' is a built-in and has no bundle.",
        )

    # Custom rows always have a bundle (XOR check constraint), but guard
    # explicitly rather than assert so a corrupt row fails loud, not silent.
    if skill.bundle_file_id is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Skill '{skill.slug}' has no bundle to replace.",
        )

    old_bundle_file_id = skill.bundle_file_id
    skill.bundle_file_id = new_bundle_file_id
    skill.bundle_sha256 = new_bundle_sha256
    skill.name = new_name
    skill.description = new_description
    db_session.flush()
    return old_bundle_file_id


def update_skill_fields(
    *,
    skill: Skill,
    db_session: Session,
    is_public: bool | None = None,
    public_permission: SkillSharePermission | None = None,
    enabled: bool | None = None,
) -> Skill:
    if is_public is not None:
        skill.is_public = is_public
    if public_permission is not None:
        skill.public_permission = public_permission
    if enabled is not None:
        skill.enabled = enabled
    db_session.flush()
    return skill


def replace_skill_shares(
    *,
    skill: Skill,
    db_session: Session,
    user_shares: Mapping[UUID, SkillSharePermission] | None = None,
    group_shares: Mapping[int, SkillSharePermission] | None = None,
) -> None:
    if user_shares is not None:
        db_session.execute(delete(Skill__User).where(Skill__User.skill_id == skill.id))
        for user_id, permission in user_shares.items():
            db_session.add(
                Skill__User(skill_id=skill.id, user_id=user_id, permission=permission)
            )

    if group_shares is not None:
        db_session.execute(
            delete(Skill__UserGroup).where(Skill__UserGroup.skill_id == skill.id)
        )
        for group_id, permission in group_shares.items():
            db_session.add(
                Skill__UserGroup(
                    skill_id=skill.id,
                    user_group_id=group_id,
                    permission=permission,
                )
            )

    try:
        db_session.flush()
    except IntegrityError as e:
        if is_fk_violation(e):
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "One or more share targets do not exist.",
            ) from e
        raise


def transfer_skill_ownership(
    *,
    skill: Skill,
    new_owner_user_id: UUID,
    db_session: Session,
) -> None:
    previous_owner_user_id = skill.author_user_id
    skill.author_user_id = new_owner_user_id

    db_session.execute(
        delete(Skill__User).where(
            Skill__User.skill_id == skill.id,
            Skill__User.user_id == new_owner_user_id,
        )
    )

    if (
        previous_owner_user_id is not None
        and previous_owner_user_id != new_owner_user_id
    ):
        existing_share = db_session.scalar(
            select(Skill__User).where(
                Skill__User.skill_id == skill.id,
                Skill__User.user_id == previous_owner_user_id,
            )
        )
        if existing_share is not None:
            existing_share.permission = SkillSharePermission.EDITOR
        else:
            db_session.add(
                Skill__User(
                    skill_id=skill.id,
                    user_id=previous_owner_user_id,
                    permission=SkillSharePermission.EDITOR,
                )
            )

    db_session.flush()


def delete_skill(skill: Skill, db_session: Session) -> str | None:
    """Hard-delete a skill and return its `bundle_file_id` for caller cleanup."""
    bundle_file_id = skill.bundle_file_id
    db_session.delete(skill)
    db_session.flush()
    return bundle_file_id
