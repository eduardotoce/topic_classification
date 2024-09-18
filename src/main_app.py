import streamlit as st
from streamlit_option_menu import option_menu

from core.demo_interface.customer_support_page.frontend.chatbot_assistant import chatbot_assistant_main
from core.demo_interface.specialized_agent_page.frontend.specialized_agent import specialized_agent_main
from core.utils.graphing_functions import local_css

st.set_page_config(
    page_title="ADP Chatbot Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Apply global styling to all pages
    local_css("static/style/styling.css")

    st.write(
        f"""<style>div.block-container{{padding-top:0rem;
                                             padding-left:1.5rem;
                                             padding-right:1.5rem;
                                             padding-bottom:0rem;}}</style>""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns((1, 9))

    with c1:
        st.image("static/images/adp_slogan.png")
        st.write("")

    page_names = ["ADP Chatbot Assistant", "ADP Specialized Agent page"]

    with c2:
        page_choice = option_menu(
            menu_title=None,
            options=page_names,
            orientation="horizontal",
            icons=[
                "fan",
                "file-earmark-bar-graph-fill",
                "gear",
                "list-task",
            ],
            menu_icon="cast",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "white",
                    "border-color": "#BFBFBF",
                    "border-width": "1px",
                    "border-style": "solid",
                },
                "nav-link": {
                    "font-size": "15px",
                    "margin": "1px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#BFBFBF", "color": "white"},
            },
        )

    st.session_state.curr_page = page_names.index(page_choice)
    st_app = [chatbot_assistant_main, specialized_agent_main]

    page_turning_function = st_app[st.session_state.curr_page]

    page_turning_function(st.session_state)


if __name__ == "__main__":
    main()
