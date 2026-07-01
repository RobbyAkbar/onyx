import secrets

from fastapi import APIRouter
from fastapi import Depends
from fastapi_users import exceptions
from pydantic import BaseModel
from pydantic import EmailStr

from onyx.auth.permissions import require_permission
from onyx.auth.schemas import UserCreate
from onyx.auth.schemas import UserRole
from onyx.auth.users import auth_backend
from onyx.auth.users import get_user_manager
from onyx.auth.users import UserManager
from onyx.configs.app_configs import SESSION_EXPIRE_TIME_SECONDS
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.utils.logger import setup_logger

logger = setup_logger()

router = APIRouter(prefix="/admin/sso")


class SsoExchangeRequest(BaseModel):
    email: EmailStr


class SsoExchangeResponse(BaseModel):
    access_token: str
    expires_in: int


@router.post("/exchange")
async def sso_exchange(
    payload: SsoExchangeRequest,
    # Admin-key gate: the caller (e.g. Microo.RestAPI) proves it is a trusted
    # server by presenting an admin API key. This endpoint is effectively
    # impersonation-by-email, so it must never be reachable by end users — keep
    # it admin-only and ideally allow-list it to the trusted backend's egress.
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    user_manager: UserManager = Depends(get_user_manager),
    strategy=Depends(auth_backend.get_strategy),
) -> SsoExchangeResponse:
    """Server-to-server identity bridge for an external SSO front-end.

    Given an already-authenticated end user's email (trust established via the
    admin API key), return an Onyx-issued bearer token for the matching real,
    email-keyed user, provisioning the user on first call. The token is minted by
    the active auth strategy (postgres / redis / jwt) via ``write_token``, so it
    is the exact same stateful session token the cookie + mobile backends issue
    and authenticates on every API route through the mobile (Bearer) backend.
    """
    email = payload.email

    try:
        user = await user_manager.get_by_email(email)
    except exceptions.UserNotExists:
        user = await user_manager.create(
            UserCreate(
                email=email,
                # Random unusable password — this user signs in via SSO only.
                password=secrets.token_urlsafe(32),
                role=UserRole.BASIC,
                is_verified=True,
            ),
            safe=False,
        )
        logger.info(f"[sso-exchange] provisioned new Onyx user for {email}")

    token = await strategy.write_token(user)
    return SsoExchangeResponse(
        access_token=token,
        expires_in=SESSION_EXPIRE_TIME_SECONDS,
    )
