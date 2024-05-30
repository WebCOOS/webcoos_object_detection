
import os
import logging
from typing import Annotated
from fastapi import Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.exceptions import HTTPException
from dataclasses import dataclass


@dataclass
class TokenGroupAssetTuple:
    token:  str
    group:  str
    asset:  str


logger = logging.getLogger( __name__ )

# Auth / token schemes
authorization_token_scheme = APIKeyHeader(
    name='Authorization',
    auto_error=False
)

# Attempt to read and parse allowed bearer tokens from the
# AUTHORIZED_BEARER_TOKENS environment variable.
AUTHORIZED_BEARER_TOKENS = os.environ.get(
    "AUTHORIZED_BEARER_TOKENS",
    None
)

if AUTHORIZED_BEARER_TOKENS and isinstance( AUTHORIZED_BEARER_TOKENS, str ):

    AUTHORIZED_BEARER_TOKENS = list( [
        str( x ).strip().lower()
        for x in AUTHORIZED_BEARER_TOKENS.split(",")
        if str( x ).strip().lower()
    ] )

else:
    AUTHORIZED_BEARER_TOKENS = []

if not AUTHORIZED_BEARER_TOKENS:
    logger.warning(
        "AUTHORIZED_BEARER_TOKENS is unset/empty, this will disable any "
        "group/asset-specific metrics from being recorded"
    )

def get_token_group_asset_tuple(
    authorization: Annotated[str, Depends( authorization_token_scheme )],
    x_axds_group: Annotated[str, Header()] = None,
    x_axds_asset: Annotated[str, Header()] = None,
):
    token = None
    group = None
    asset = None

    if authorization and isinstance( authorization, str ):

        ( bearer, _token ) = get_authorization_scheme_param( authorization )

        if bearer.lower() != 'bearer':
            raise HTTPException(
                400,
                "Authorization header must be of format 'Bearer [token]'"
            )

        if _token:
            token = _token

    if (
        x_axds_group and
        isinstance( x_axds_group, str )
     ):
        group = x_axds_group

    if (
        x_axds_asset and
        isinstance( x_axds_asset, str )
     ):
        asset = x_axds_asset

    return TokenGroupAssetTuple(
        token,
        group,
        asset
    )

def assert_auth_token_group_and_asset( tga: TokenGroupAssetTuple ):

    assert tga is not None and isinstance( tga, TokenGroupAssetTuple )

    if tga.group and tga.asset:

        if not tga.token:

            raise HTTPException(
                403,
                (
                    "group/asset headers sent, but no authorization sent"
                )
            )

        _token = str( tga.token ).strip().lower()

        if _token not in AUTHORIZED_BEARER_TOKENS:

            raise HTTPException(
                403,
                (
                    "authorization/group/asset headers sent, but token not "
                    "recognized"
                )
            )

    return
