import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* ブラウザ全体のスクロールバーを消す */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
    }

    /* メインエリアの余白（パディング）を小さくして画面を広く使う */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* 右側の上下カードの高さを固定して、内容が増えてもカード内でスクロールさせる */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        max-height: 40vh;
        overflow-y: auto;
    }

    /* スライダー周辺の余白を小さくする */
    div[data-testid="stSlider"] {
        margin-top: -10px !important;
        margin-bottom: -10px !important;
    }
</style>
""", unsafe_allow_html=True)

#ガンマsRGB→LinearRGB変換
def srgb_to_linear(srgb):
    srgb = np.asarray(srgb, dtype=float)

    #入力が8bitの場合は正規化
    if np.any(srgb > 1.0):
        srgb = srgb / 255.0

    # 安全のため、0.0未満や1.0を超える値を範囲内に収める（NaN対策）
    srgb = np.clip(srgb, 0.0, 1.0)

    #sRGB公式の条件分岐(参考：https://en.wikipedia.org/wiki/SRGB)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )

    return linear

#行列の定義
#リニアsRGB→CIE XYZ行列(参考：https://fujiwaratko.sakura.ne.jp/infosci/colorspace/colorspace2.html)
M_srgb_to_xyz = np.array([
    [0.412391, 0.357584, 0.180481],
    [0.212639, 0.715169, 0.072192],
    [0.019331, 0.119195, 0.950532]
])

#XYZ→LMS行列(参考：Fairchild, Mark D. (2005). Color Appearance Models)
M_xyz_to_lms = np.array([
    [0.3897, 0.6890, -0.0787],
    [-0.2298, 1.1834, 0.0464],
    [0.0000, 0.0000, 1.0000]
])

#行列のまとめ
M_srgb_to_lms = M_xyz_to_lms @ M_srgb_to_xyz
M_lms_to_srgb = np.linalg.inv(M_srgb_to_lms) #逆行列

# 基準グレー(0.5　0.5　0.5) の LMS 値を算出
gray_lin = srgb_to_linear([0.5, 0.5, 0.5])
lms_gray = M_srgb_to_lms @ gray_lin

#θとΦを定義
phis = np.linspace(0, 2 * np.pi, 360)
thetas = np.linspace(-np.pi / 2, np.pi / 2, 90)

#半径1のDKL空間とsRGBの変換行列
L0, M0, S0 = lms_gray[0], lms_gray[1], lms_gray[2]

ru_2 = np.sqrt(2)
ru_3 = np.sqrt(3)
ru_LM = np.sqrt(L0**2 + M0**2)

#LMS→DKLの変換行列(参考：https://pooneil.sakura.ne.jp/dkl3.pdf)
M_lms_to_dkl = (1.0 / (L0 + M0)) * np.array([
    #L-M(x)
    [ru_LM / L0, -ru_LM / M0, 0],

    #S-(L+M)(y)
    [-1, -1, (L0 + M0) / S0],

    #L+M(Z)
    [ru_3, ru_3, 0]
])

M_dkl_to_lms = np.linalg.inv(M_lms_to_dkl)

#sRGBから半径1のDKL空間への変換関数
def rgb_to_dkl(rgb_0to255):
    """8bit RGB [R, G, B] から DKL 座標を計算"""
    rgb_norm = np.array(rgb_0to255) / 255.0
    
    # ガンマ解除
    rgb_lin = srgb_to_linear(rgb_norm)
    
    # 対象色の LMS 値を算出
    lms = M_srgb_to_lms @ rgb_lin
    
    # 白からの差分 (dL, dM, dS)
    dLMS = lms - lms_gray
    
    # DKL 座標 (LUM, L-M, S)
    dkl = M_lms_to_dkl @ dLMS

    return dkl

#半径1のDKL空間からsRGBへの変関数
def dkl_to_rgb(dkl):
    dLMS = M_dkl_to_lms @ dkl

    lms = lms_gray + dLMS

    rgb_lin = M_lms_to_srgb @ lms

    rgb_255 = np.clip(rgb_lin * 255.0, 0, 255).astype(np.uint8)

    return rgb_255

#ディスプレイの色域をクリッピング
r_limits_all = []
for phi in phis:
    for theta in thetas:
        dkl_dir = np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(theta) * np.sin(phi),
            np.sin(theta)
        ])

        drgb_lin = M_lms_to_srgb @ (M_dkl_to_lms @ dkl_dir)

        r_dir_limits = []

        for i in range(3):
            if drgb_lin[i] > 1e-8:
                r_dir_limits.append((1.0 - gray_lin[i]) / drgb_lin[i])
            elif drgb_lin[i] < -1e-8:
                r_dir_limits.append((0.0 - gray_lin[i]) / drgb_lin[i])

        if r_dir_limits:
            r_limits_all.append(min(r_dir_limits))

r_min = min(r_limits_all)

#グラフの作成関数
def create_dkl_base_figure(axis_config):
    fig = go.Figure()

    # 共通の3軸を追加
    fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='red', width=4), text=['','L-M'], textposition='top center', textfont=dict(color='white', size=12), name='L-M 軸'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines+text', line=dict(color='blue', width=4), text=['','S-(L+M)'], textposition='top center', textfont=dict(color='white', size=12), name='S-(L+M) 軸'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines+text', line=dict(color='gray', width=4), text=['','L+M'], textposition='top center', textfont=dict(color='white', size=12), name='Luminance 軸'))

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=axis_config,
            yaxis=axis_config,
            zaxis=axis_config,
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=220,
        scene_camera=dict(
            eye=dict(x=0.9, y=0.9, z=0.8)
        )
    )
    return fig

# 共通で使うPlotly configパラメータ
chart_config = {'scrollZoom': False, 'displayModeBar': False}

# 軸のクリーン設定
axis_clean_config = dict(
    showgrid=False,
    showbackground=False,
    showticklabels=False,
    zeroline=False,
    visible=False,
    range=[-1.1, 1.1]
)

#ページレイアウト
col1, col2 = st.columns([3, 2])

#実験シミュレーション
with col1:
    st.markdown("<h4 style='text-align: center;'>Experimental simulation</h4>", unsafe_allow_html=True)

    # 系列名を中央に小さく配置
    _, series_col, _ = st.columns([3, 2, 3])

    with series_col:
        series_name = st.selectbox(
            "評価する印象",
            ["好きなー嫌いな", "美しいー見苦しい", "洗練されたー野暮ったい", "心地よいー不快な", "鮮やかーくすんだ", "派手ー地味", "暖かいー冷たい", "陽気なー陰気な", "硬いー柔らかい", "強いー弱い", "重いー軽い", "複雑ー単純", "均一ー不均一"],
            label_visibility="collapsed"
        )

    # 片矢印
    st.markdown("""
    <div style="
        width: 100%;
        padding: 0px 20px 10px 20px;
        box-sizing: border-box;
    ">
        <div style="
            width: 100%;
            height: 2px;
            background-color: white;
            position: relative;
        ">
            <div style="
                position: absolute;
                right: -1px;
                top: -6px;
                width: 0;
                height: 0;
                border-top: 7px solid transparent;
                border-bottom: 7px solid transparent;
                border-left: 12px solid white;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    Experiment_container = st.empty()

with col2:
    
    st.markdown("<h4 style='text-align: center;'>Color settings</h4>", unsafe_allow_html=True)

    with st.container(border=True):

        top_left, top_right = st.columns([1, 1])

        with top_left:
            st.write("Setting the average color")

            average_dkl_r_N = st.slider("r", 0.0, 1.0, 0.3, step=0.05)
            average_dkl_phi_d = st.slider("Φ", 0, 360, 30, step=1)
            average_dkl_theta_d = st.slider("θ", -90, 90, 30, step=1)

            Contrast_dis = st.empty()

            #RGBの色域の最大値を1としたrから目の感度の最大値を1にしたｒへの変換
            average_dkl_r = r_min * average_dkl_r_N

            average_dkl_phi = np.deg2rad(average_dkl_phi_d)
            average_dkl_theta = np.deg2rad(average_dkl_theta_d)

            if average_dkl_theta_d == 90:
                # +L+M方向
                M_average_dkl = np.array([0.0, 0.0, 1.0])

            elif average_dkl_theta_d == -90:
                # -L-M方向
                M_average_dkl = np.array([0.0, 0.0, -1.0])

            else:
                M_average_dkl = np.array([
                    #L-M
                    np.cos(average_dkl_theta) * np.cos(average_dkl_phi),
                    #S-(L+M)
                    np.cos(average_dkl_theta) * np.sin(average_dkl_phi),
                    #L+M
                    np.sin(average_dkl_theta)
                ])

            #計算用DKL座標
            average_dkl = average_dkl_r * M_average_dkl

            #グラフ用
            graph_average_dkl = average_dkl_r_N * M_average_dkl

            color_rgb_255 = dkl_to_rgb(average_dkl)

            color_rgb_str = f"rgb({color_rgb_255[0]}, {color_rgb_255[1]}, {color_rgb_255[2]})"


        with top_right:
            st.write("DKL Color Space")
            st.caption(f"R:{color_rgb_255[0]} G:{color_rgb_255[1]} B:{color_rgb_255[2]}")
            fig1 = create_dkl_base_figure(axis_clean_config)
            
            # DKL 軸の描画, 球フレームの描画
            r = 1.0
            for phi_value in np.linspace(0, 2 * np.pi, 12, endpoint=False):

                theta_line = np.linspace(-np.pi / 2, np.pi / 2, 100)

                x_line = r * np.cos(theta_line) * np.cos(phi_value)
                y_line = r * np.cos(theta_line) * np.sin(phi_value)
                z_line = r * np.sin(theta_line)

                fig1.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(
                        color='white',
                        width=1
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

            for theta_value in np.linspace(
                -np.pi / 2,
                np.pi / 2,
                7
            ):
                phi_line = np.linspace(0, 2 * np.pi, 200)

                x_line = r * np.cos(theta_value) * np.cos(phi_line)
                y_line = r * np.cos(theta_value) * np.sin(phi_line)
                z_line = np.full_like(
                    phi_line,
                    r * np.sin(theta_value)
                )

                fig1.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(
                        color='white',
                        width=1
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

            # 現在選択されている点
            fig1.add_trace(go.Scatter3d(
                x=[graph_average_dkl[0]], y=[graph_average_dkl[1]], z=[graph_average_dkl[2]],
                mode='markers',
                marker=dict(size=8, color=color_rgb_str, symbol='circle'),
                name='現在値'
            ))
       
            st.plotly_chart(fig1, use_container_width=True, config=chart_config)

    with st.container(border=True):
        # テクスチャを構成する2色のコントラスト
        # 平均色を中心として、そこから一定距離 r_contrast 離れた位置に2色を配置
        r_contrast = r_min * (1.0 - average_dkl_r_N)

        # 2方向を1組とした13種類の方向
        V_contrast = np.array([
            # 軸方向 3組
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],

            # xy平面 2組
            [1/ru_2,  1/ru_2, 0],
            [1/ru_2, -1/ru_2, 0],

            # yz平面 2組
            [0, 1/ru_2,  1/ru_2],
            [0, 1/ru_2, -1/ru_2],

            # xz平面 2組
            [1/ru_2, 0,  1/ru_2],
            [1/ru_2, 0, -1/ru_2],

            # 立方体の頂点方向 4組
            [1/ru_3,  1/ru_3,  1/ru_3],
            [-1/ru_3, 1/ru_3,  1/ru_3],
            [1/ru_3, -1/ru_3,  1/ru_3],
            [1/ru_3,  1/ru_3, -1/ru_3]
        ])

        # 26色を格納
        contrast_dkl_list = []
        contrast_rgb_list = []

        for v in V_contrast:

            # 平均色から +方向
            dkl_plus = average_dkl + r_contrast * v

            # 平均色から -方向
            dkl_minus = average_dkl - r_contrast * v

            # DKL座標を保存
            contrast_dkl_list.append(dkl_plus)
            contrast_dkl_list.append(dkl_minus)

            # RGBに変換
            rgb_plus = dkl_to_rgb(dkl_plus)
            rgb_minus = dkl_to_rgb(dkl_minus)

            #streamlitの形式に変換
            rgb_plus_s = f"rgb({rgb_plus[0]}, {rgb_plus[1]}, {rgb_plus[2]})"
            rgb_minus_s = f"rgb({rgb_minus[0]}, {rgb_minus[1]}, {rgb_minus[2]})"

            # RGBを保存
            contrast_rgb_list.append(rgb_plus_s)
            contrast_rgb_list.append(rgb_minus_s)

        bottom_left, bottom_right = st.columns([1, 1])

        with bottom_right:
            st.write("Color combinations")
            Contrast_dis.markdown(
                f"contrast_max：R = {r_contrast:.3f}",
                unsafe_allow_html=True
            )

            fig2 = create_dkl_base_figure(axis_clean_config)

            #組み合わせの色立体を表示
            for com_num, com in enumerate(V_contrast):
                dkl_plus_N = com
                dkl_minus_N = -com

                # + と - を結ぶ線
                fig2.add_trace(go.Scatter3d(
                    x=[dkl_minus_N[0], dkl_plus_N[0]],
                    y=[dkl_minus_N[1], dkl_plus_N[1]],
                    z=[dkl_minus_N[2], dkl_plus_N[2]],
                    mode='lines+text',
                    text=f"{com_num}",
                    line=dict(
                        color='white',
                        width=1
                    ),
                    showlegend=False
                ))

                fig2.add_trace(go.Scatter3d(
                    x=[dkl_plus_N[0]], y=[dkl_plus_N[1]], z=[dkl_plus_N[2]],
                    mode='markers',
                    marker=dict(size=7, color=contrast_rgb_list[com_num * 2], symbol='circle'),
                    name='+'
                ))

                fig2.add_trace(go.Scatter3d(
                    x=[dkl_minus_N[0]], y=[dkl_minus_N[1]], z=[dkl_minus_N[2]],
                    mode='markers',
                    marker=dict(size=7, color=contrast_rgb_list[com_num * 2 + 1], symbol='circle'),
                    name='-'
                ))

            st.plotly_chart(fig2, use_container_width=True, config=chart_config)

        with bottom_left:
            st.write("Texture image")

            preview_container = st.empty()

            texture_number = st.slider("texture number", 1, 13, 1, step=1)
            texture_size = st.slider("pixel size", 1, 100, 20, step=1)

            #テクスチャプレビュー用のHTML
            preview_html = f"""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 10px 0;">
                <div style="
                    width: 80px;
                    height: 80px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                    background: repeating-conic-gradient({contrast_rgb_list[texture_number * 2 - 2]} 0% 25%, {contrast_rgb_list[texture_number * 2 - 1]} 0% 50%) 50% / {texture_size}px {texture_size}px;
                ">
                </div>
            </div>
            """
            preview_container.markdown(preview_html, unsafe_allow_html=True)

#HTML,CSS,JavaScriptの構築
drag_html = f"""
<style>
    .container {{
        width: 100%;
        height: 500px;
        position: relative;

        overflow: hidden;
    }}
    /* 共通のボックススタイル */
    .box {{
        width: 100px;
        height: 100px;
        
        border-radius: 1px;
        position: absolute;

        cursor: grab;
        user-select: none;
        color: white;
        font-size: 0.7rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s;
    }}
    .box:active {{
        cursor: grabbing;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        z-index: 10; /* ドラッグ中のボックスを一番前に出す */
    }}
    
    /* 各ボックス個別の色と初期位置 */
    #box1 {{ background: repeating-conic-gradient(
        {contrast_rgb_list[0]} 0% 25%,
        {contrast_rgb_list[1]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box2 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[2]} 0% 25%,
            {contrast_rgb_list[3]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box3 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[4]} 0% 25%,
            {contrast_rgb_list[5]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box4 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[6]} 0% 25%,
            {contrast_rgb_list[7]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box5 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[8]} 0% 25%,
            {contrast_rgb_list[9]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box6 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[10]} 0% 25%,
            {contrast_rgb_list[11]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}  
    #box7 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[12]} 0% 25%,
            {contrast_rgb_list[13]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box8 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[14]} 0% 25%,
            {contrast_rgb_list[15]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box9 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[16]} 0% 25%,
            {contrast_rgb_list[17]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box10 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[18]} 0% 25%,
            {contrast_rgb_list[19]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box11 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[20]} 0% 25%,
            {contrast_rgb_list[21]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box12 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[22]} 0% 25%,
            {contrast_rgb_list[23]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box13 {{ background: repeating-conic-gradient(
            {contrast_rgb_list[24]} 0% 25%,
            {contrast_rgb_list[25]} 0% 50%
    ) 50% / {texture_size}px {texture_size}px; }}
    #box14 {{ background-color: {color_rgb_str}; height: 100px; width: 100px; }}
       
</style>

<div class="container" id="canvas">
    <div class="box" id="box1">1</div>
    <div class="box" id="box2">2</div>
    <div class="box" id="box3">3</div>
    <div class="box" id="box4">4</div>
    <div class="box" id="box5">5</div>
    <div class="box" id="box6">6</div>
    <div class="box" id="box7">7</div>
    <div class="box" id="box8">8</div>
    <div class="box" id="box9">9</div>
    <div class="box" id="box10">10</div>
    <div class="box" id="box11">11</div>
    <div class="box" id="box12">12</div>
    <div class="box" id="box13">13</div>
    <div class="box" id="box14"></div>
</div>

<script>
    // 先に要素を取得
    const canvas = document.getElementById('canvas');
    const boxes = document.querySelectorAll('.box');


    // =========================
    // 初期配置
    // =========================

    function arrangeBoxes() {{
        const gap = 20;
        const boxWidth = 100;
        const boxHeight = 100;
        const margin = 20;

        const canvasWidth = canvas.clientWidth;

        // 横に何個並べられるか
        const columns = Math.max(
            1,
            Math.floor(
                (canvasWidth - margin * 2 + gap)
                / (boxWidth + gap)
            )
        );

        boxes.forEach((box, index) => {{
            const col = index % columns;
            const row = Math.floor(index / columns);

            box.style.left =
                (margin + col * (boxWidth + gap)) + 'px';

            box.style.top =
                (margin + row * (boxHeight + gap)) + 'px';
        }});
    }}


    // 初期配置
    arrangeBoxes();


    // =========================
    // ドラッグ処理
    // =========================

    let activeBox = null;
    let offsetX = 0;
    let offsetY = 0;

    boxes.forEach(box => {{
        box.addEventListener('mousedown', (e) => {{
            activeBox = box;

            const canvasRect = canvas.getBoundingClientRect();

            offsetX =
                e.clientX - canvasRect.left - box.offsetLeft;

            offsetY =
                e.clientY - canvasRect.top - box.offsetTop;

            box.style.zIndex = 10;
        }});
    }});



    document.addEventListener('mousemove', (e) => {{
        if (!activeBox) return;

        const canvasRect = canvas.getBoundingClientRect();

        let x =
            e.clientX - canvasRect.left - offsetX;

        let y =
            e.clientY - canvasRect.top - offsetY;

        // 枠外にはみ出さない
        const maxX =
            canvas.clientWidth - activeBox.clientWidth;

        const maxY =
            canvas.clientHeight - activeBox.clientHeight;

        x = Math.max(0, Math.min(x, maxX));
        y = Math.max(0, Math.min(y, maxY));

        activeBox.style.left = x + 'px';
        activeBox.style.top = y + 'px';
    }});


    document.addEventListener('mouseup', () => {{
        if (activeBox) {{
            activeBox.style.zIndex = '';
        }}

        activeBox = null;
    }});
</script>
"""

with Experiment_container:
    components.html(drag_html, height=600)