"""Logging/Logout UI"""

import streamlit as st
from .simple_auth import is_authenticated, current_user, logout_button

def render_login_sidebar(
    title: str = "Account",
    subtitle: str = "Sign in to continue",
    provider: str = "auth0",
) -> None:
    """Auth UI that lives in the Streamlit sidebar."""
    with st.sidebar:
        st.markdown(f"### {title}")

        if not is_authenticated():  
            if st.button("Log in", type="primary", use_container_width=True):
                st.login(provider)
            st.caption(subtitle)
        else:
            u = current_user()

            # Compact user chip
            row = st.container()
            with row:
                c1, c2 = st.columns([1, 3])
                with c1:
                    avatar = u.get("picture")
                    if avatar:
                        st.image(avatar, width=32)
                with c2:
                    st.markdown(f"**{u.get('name') or 'User'}**")
                    if u.get("email"):
                        st.caption(u["email"])

            logout_button("Log out")  
