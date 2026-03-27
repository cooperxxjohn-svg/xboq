"""
Streamlit login UI component — renders login form and manages session state.

Usage in demo_page.py:
    from src.auth.login_ui import render_auth_gate, get_current_user

    user = render_auth_gate()
    if user:
        # rest of the app
        ...
"""
import streamlit as st
from typing import Optional, Dict
from .supabase_client import is_configured, sign_in, sign_out, get_user


# ── Design tokens (match xBOQ.ai website exactly) ───────────────────────────
_ACCENT       = "#7c3aed"   # violet-600
_ACCENT_LIGHT = "#a78bfa"   # violet-400
_ACCENT_SOFT  = "#c4b5fd"   # violet-300
_BG_BASE      = "#09090b"   # zinc-950
_BG_CARD      = "#111113"   # zinc-900
_BORDER       = "rgba(255,255,255,0.08)"
_TEXT_MUTED   = "#71717a"   # zinc-500

# ── Login page CSS ───────────────────────────────────────────────────────────
_LOGIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: #09090b;
    font-family: 'Inter', -apple-system, sans-serif;
}

/* ── Full-screen login wrapper ── */
.xboq-login-bg {
    position: fixed;
    inset: 0;
    background: #09090b;
    overflow: hidden;
    z-index: 0;
}

/* Glow blob 1 — top-left violet */
.xboq-login-bg::before {
    content: '';
    position: absolute;
    top: -180px; left: -120px;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.35) 0%, transparent 65%);
    animation: glow-drift 8s ease-in-out infinite alternate;
    pointer-events: none;
}

/* Glow blob 2 — bottom-right indigo */
.xboq-login-bg::after {
    content: '';
    position: absolute;
    bottom: -150px; right: -100px;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,40,220,0.28) 0%, transparent 65%);
    animation: glow-drift 10s ease-in-out infinite alternate-reverse;
    pointer-events: none;
}

@keyframes glow-drift {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(40px, 30px) scale(1.08); }
}

/* ── Dot particle grid ── */
.xboq-particles {
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle, rgba(124,58,237,0.18) 1px, transparent 1px),
        radial-gradient(circle, rgba(167,139,250,0.1) 1px, transparent 1px);
    background-size: 48px 48px, 96px 96px;
    background-position: 0 0, 24px 24px;
    pointer-events: none;
    z-index: 1;
    animation: particle-fade 6s ease-in-out infinite alternate;
}
@keyframes particle-fade {
    0%   { opacity: 0.6; }
    100% { opacity: 1; }
}

/* ── Login card ── */
.xboq-card-wrap {
    position: relative;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 2rem 1rem;
}

.xboq-login-card {
    width: 100%;
    max-width: 420px;
    background: rgba(17,17,19,0.85);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow:
        0 0 0 1px rgba(124,58,237,0.1),
        0 32px 64px rgba(0,0,0,0.5),
        0 0 120px rgba(124,58,237,0.08);
}

/* ── Logo area ── */
.xboq-logo-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.75rem;
}
.xboq-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 800; color: white;
    box-shadow: 0 0 20px rgba(124,58,237,0.5);
    flex-shrink: 0;
}
.xboq-logo-text {
    font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, #e4e4e7 30%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

/* ── Headline ── */
.xboq-login-headline {
    font-size: 1.5rem; font-weight: 700;
    color: #e4e4e7;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.xboq-login-sub {
    font-size: 0.875rem; color: #71717a;
    margin-bottom: 1.75rem;
    line-height: 1.5;
}

/* ── Form inputs ── */
.xboq-login-card .stTextInput label,
.xboq-login-card label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #a1a1aa !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 0.3rem !important;
}
.xboq-login-card .stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e4e4e7 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.9rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.xboq-login-card .stTextInput > div > div > input:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

/* ── Submit button ── */
.xboq-login-card .stFormSubmitButton > button,
.xboq-login-card .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.35) !important;
    margin-top: 0.5rem !important;
}
.xboq-login-card .stFormSubmitButton > button:hover,
.xboq-login-card .stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
    box-shadow: 0 6px 32px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider inside form ── */
.xboq-login-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.25rem 0;
}

/* ── Error/Success alerts ── */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

/* ── Stats strip below card ── */
.xboq-stats-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 2rem;
    padding: 1rem 0;
}
.xboq-stat {
    text-align: center;
}
.xboq-stat-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e4e4e7;
    display: block;
    line-height: 1.1;
}
.xboq-stat-value.accent { color: #a78bfa; }
.xboq-stat-label {
    font-size: 0.68rem;
    color: #52525b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
    margin-top: 0.15rem;
}

/* ── Tenant badge ── */
.xboq-tenant-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem; font-weight: 600;
    color: #a78bfa;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
}
.xboq-tenant-dot {
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 6px #4ade80;
}

/* ── Spinner color override ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Remove Streamlit padding on login page ── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
</style>
"""

# ── Particle canvas JS (optional floating dots layer) ───────────────────────
_PARTICLE_JS = """
<canvas id="xboq-particles-canvas" style="
    position:fixed; inset:0; z-index:2; pointer-events:none;
    width:100%; height:100%;
"></canvas>
<script>
(function(){
    var c = document.getElementById('xboq-particles-canvas');
    if(!c) return;
    var ctx = c.getContext('2d');
    var W = window.innerWidth, H = window.innerHeight;
    c.width = W; c.height = H;
    var dots = [];
    for(var i=0;i<120;i++){
        dots.push({
            x: Math.random()*W, y: Math.random()*H,
            r: Math.random()*1.5+0.3,
            a: Math.random(),
            da: (Math.random()-0.5)*0.004,
            dx: (Math.random()-0.5)*0.15,
            dy: (Math.random()-0.5)*0.15,
        });
    }
    function draw(){
        ctx.clearRect(0,0,W,H);
        dots.forEach(function(d){
            d.x += d.dx; d.y += d.dy; d.a += d.da;
            if(d.x<0) d.x=W; if(d.x>W) d.x=0;
            if(d.y<0) d.y=H; if(d.y>H) d.y=0;
            if(d.a<0.1 || d.a>0.9) d.da=-d.da;
            ctx.beginPath();
            ctx.arc(d.x, d.y, d.r, 0, 2*Math.PI);
            var violet = Math.random()>0.5;
            ctx.fillStyle = violet
                ? 'rgba(167,139,250,'+d.a+')'
                : 'rgba(196,181,253,'+d.a+')';
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }
    draw();
    window.addEventListener('resize', function(){
        W = window.innerWidth; H = window.innerHeight;
        c.width = W; c.height = H;
    });
})();
</script>
"""


def get_current_user() -> Optional[Dict]:
    """Return current user from session state, or None."""
    token = st.session_state.get("auth_token")
    if not token:
        return None
    user = get_user(token)
    if not user:
        st.session_state.pop("auth_token", None)
        st.session_state.pop("auth_user", None)
    return user


def render_auth_gate() -> Optional[Dict]:
    """
    Auth gate — currently bypassed (login disabled for demo).
    Re-enable by removing the early return below.
    """
    # Login disabled for demo — always return guest
    return {"id": "guest", "email": "guest@local", "org_id": "local"}

    # pylint: disable=unreachable
    if not is_configured():
        return {"id": "guest", "email": "guest@local", "org_id": "local"}

    user = get_current_user()
    if user:
        return user

    _render_login_form()
    st.stop()
    return None


def _render_login_form():
    """Render a full-screen dark-theme login form matching xBOQ.ai."""
    # Inject CSS + background layers
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="xboq-login-bg"></div>'
        '<div class="xboq-particles"></div>',
        unsafe_allow_html=True
    )
    st.markdown(_PARTICLE_JS, unsafe_allow_html=True)

    # Center column layout
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        # Card wrapper open
        st.markdown('<div class="xboq-card-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="xboq-login-card">', unsafe_allow_html=True)

        # Logo
        st.markdown("""
        <div class="xboq-logo-row">
            <div class="xboq-logo-icon">X</div>
            <span class="xboq-logo-text">xBOQ</span>
        </div>
        """, unsafe_allow_html=True)

        # Tenant status badge
        st.markdown("""
        <div class="xboq-tenant-badge">
            <span class="xboq-tenant-dot"></span>
            Bid Engineer Platform
        </div>
        """, unsafe_allow_html=True)

        # Headline
        st.markdown("""
        <div class="xboq-login-headline">Sign in to your workspace</div>
        <div class="xboq-login-sub">
            AI-powered tender analysis for construction contractors.
        </div>
        """, unsafe_allow_html=True)

        # Form
        with st.form("xboq_login_form", clear_on_submit=False):
            email    = st.text_input("Email", placeholder="you@firm.com", label_visibility="visible")
            password = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="visible")
            submit   = st.form_submit_button("Sign In →", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                with st.spinner("Signing in…"):
                    result = sign_in(email, password)
                if result["error"]:
                    st.error(f"Login failed: {result['error']}")
                else:
                    st.session_state["auth_token"] = result["session"].access_token
                    st.session_state["auth_user"]  = result["user"]
                    st.rerun()

        # Stats strip below card
        st.markdown("""
        <hr class="xboq-login-divider"/>
        <div class="xboq-stats-strip">
            <div class="xboq-stat">
                <span class="xboq-stat-value accent">200+</span>
                <span class="xboq-stat-label">Pages parsed</span>
            </div>
            <div class="xboq-stat">
                <span class="xboq-stat-value">48</span>
                <span class="xboq-stat-label">Analysis modules</span>
            </div>
            <div class="xboq-stat">
                <span class="xboq-stat-value accent">&lt;24h</span>
                <span class="xboq-stat-label">First report</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)


def render_user_menu():
    """Render user info + sign-out button in sidebar."""
    user = get_current_user()
    if not user or user.get("id") == "guest":
        return
    with st.sidebar:
        st.markdown("---")
        st.caption(f"Signed in: {user.get('email', '')}")
        if st.button("Sign Out", use_container_width=True):
            token = st.session_state.pop("auth_token", None)
            st.session_state.pop("auth_user", None)
            if token:
                sign_out(token)
            st.rerun()
