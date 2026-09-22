import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import matplotlib.cm as cm
from scipy import signal
import os
import io
import csv
import base64
import json
import warnings
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# Hỗ trợ linh hoạt TFLite / LiteRT
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
        class TFLiteWrapper:
            Interpreter = TFLiteInterpreter
        tflite = TFLiteWrapper()
    except ImportError:
        import tensorflow.lite as tflite

# ==========================================
# CẤU HÌNH THAM SỐ TOÀN CỤC
# ==========================================
TARGET_SR = 16000
WINDOW_SIZE = 15600                 # 0.975s ở 16kHz
HOP_SIZE = 7800                     # 0.4875s (50% Overlap)
HOP_DURATION = HOP_SIZE / TARGET_SR

CONFIDENCE_THRESHOLD = 0.50
VOTING_WINDOW_COUNT = 3

MIN_ENERGY_RMS = 0.02
CREST_FACTOR_THRESH = 6.5
SUB_BASS_RATIO_THRESH = 0.85

FFT_FREQS = np.fft.rfftfreq(WINDOW_SIZE, 1.0 / TARGET_SR)
SUB_BASS_MASK = (FFT_FREQS >= 40.0) & (FFT_FREQS < 250.0)
VALID_AUDIO_MASK = (FFT_FREQS >= 40.0)

YAMNET_PATH = "yamnet_full.tflite"
HEAD_PATH = "chainsaw_head.tflite"

st.set_page_config(page_title="AI Chainsaw Edge Inspector Web", layout="wide", page_icon="🌲")

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH VÀO CACHE
# ==========================================
@st.cache_resource(show_spinner="Đang nạp chuỗi mô hình TFLite...")
def load_models():
    if not os.path.exists(YAMNET_PATH) or not os.path.exists(HEAD_PATH):
        return None, None
    
    interp_y = tflite.Interpreter(model_path=YAMNET_PATH)
    interp_y.resize_tensor_input(interp_y.get_input_details()[0]['index'], [WINDOW_SIZE])
    interp_y.allocate_tensors()

    interp_h = tflite.Interpreter(model_path=HEAD_PATH)
    interp_h.allocate_tensors()
    return interp_y, interp_h

interp_yamnet, interp_head = load_models()

# ==========================================
# 2. HÀM KIỂM ĐỊNH VẬT LÝ ÂM HỌC
# ==========================================
def check_mechanical_artifact(chunk):
    clean_chunk = chunk - np.mean(chunk)
    rms = np.sqrt(np.mean(clean_chunk**2))
    
    if rms < MIN_ENERGY_RMS:
        return False, 0.0, 0.0, float(rms)
        
    peak = np.max(np.abs(clean_chunk))
    crest_factor = peak / (rms + 1e-8)

    fft_magnitudes = np.abs(np.fft.rfft(clean_chunk))
    energy_spectrum = fft_magnitudes ** 2
    audio_energy = np.sum(energy_spectrum[VALID_AUDIO_MASK]) + 1e-8
    sub_bass_energy = np.sum(energy_spectrum[SUB_BASS_MASK])
    sub_bass_ratio = sub_bass_energy / audio_energy

    is_artifact = False
    if crest_factor > CREST_FACTOR_THRESH and sub_bass_ratio > SUB_BASS_RATIO_THRESH:
        is_artifact = True
    elif crest_factor > 9.0:
        is_artifact = True

    return is_artifact, float(crest_factor), float(sub_bass_ratio), float(rms)

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🌲 AI Chainsaw Acoustic Detection Web Portal")
st.caption("Nền tảng kiểm định & nhận dạng tiếng cưa máy bằng chuỗi 2 mô hình TFLite (YAMNet + Custom Head)")

if interp_yamnet is None or interp_head is None:
    st.error(f"❌ Không tìm thấy `{YAMNET_PATH}` hoặc `{HEAD_PATH}` trong cùng thư mục chạy script!")
    st.stop()

# Sidebar cấu hình
st.sidebar.header("⚙️ Cấu hình bộ lọc & Nhận diện")
conf_thresh = st.sidebar.slider("Ngưỡng xác suất (Confidence Threshold)", 0.10, 0.90, CONFIDENCE_THRESHOLD, 0.05)
voting_thresh = st.sidebar.slider("Ngưỡng số khung Voting liên tiếp", 1, 6, VOTING_WINDOW_COUNT, 1)
enable_artifact_filter = st.sidebar.checkbox("Bật bộ lọc chống gõ mic / va đập cơ học", True)

# Khu vực Upload file
uploaded_file = st.sidebar.file_uploader("Tải lên file âm thanh", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên một file âm thanh ở thanh bên trái để bắt đầu phân tích.")
    st.stop()

# ==========================================
# 3. TIỀN XỬ LÝ & SUY LUẬN CHUỖI 2 MODEL
# ==========================================
with st.spinner("Đang xử lý âm thanh & suy luận mô hình..."):
    # Đọc audio
    audio_bytes = uploaded_file.read()
    raw_data, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    
    if raw_data.ndim > 1:
        raw_data = np.mean(raw_data, axis=1)

    # Resample về chuẩn 16 kHz
    if orig_sr != TARGET_SR:
        target_len = int(len(raw_data) * TARGET_SR / orig_sr)
        audio_stream = signal.resample(raw_data, target_len)
    else:
        audio_stream = raw_data

    total_samples = len(audio_stream)
    duration = total_samples / TARGET_SR

    if total_samples < WINDOW_SIZE:
        st.warning(f"File âm thanh quá ngắn ({duration:.2f}s). Cần tối thiểu 0.975s để phân tích.")
        st.stop()

    # Chuẩn bị input/output pointers của TFLite
    y_in = interp_yamnet.get_input_details()[0]['index']
    y_emb = interp_yamnet.get_output_details()[1]['index']
    h_in = interp_head.get_input_details()[0]['index']
    h_out = interp_head.get_output_details()[0]['index']

    # Chạy trượt sliding window
    logs = []
    consecutive_flags = 0
    frame_idx = 0
    total_alerts = 0

    for start_idx in range(0, total_samples - WINDOW_SIZE + 1, HOP_SIZE):
        frame_idx += 1
        chunk = audio_stream[start_idx : start_idx + WINDOW_SIZE]
        time_offset = (frame_idx - 1) * HOP_DURATION

        is_artifact, cf, sub_b, rms = check_mechanical_artifact(chunk)
        if not enable_artifact_filter:
            is_artifact = False

        score = 0.0
        is_alert = 0

        if is_artifact:
            consecutive_flags = 0
        else:
            interp_yamnet.set_tensor(y_in, chunk)
            interp_yamnet.invoke()
            emb = interp_yamnet.get_tensor(y_emb)
            emb_2d = np.reshape(emb, (1, 1024)).astype(np.float32)

            interp_head.set_tensor(h_in, emb_2d)
            interp_head.invoke()
            score = float(interp_head.get_tensor(h_out)[0][0])

            if score >= conf_thresh:
                consecutive_flags += 1
            else:
                consecutive_flags = 0

            if consecutive_flags >= voting_thresh:
                is_alert = 1
                if consecutive_flags == voting_thresh:
                    total_alerts += 1

        logs.append({
            "frame_idx": frame_idx,
            "t": round(time_offset, 4),
            "rms": round(rms, 4),
            "cf": round(cf, 2),
            "sub_b": round(sub_b, 4),
            "artifact": 1 if is_artifact else 0,
            "score": round(score, 4),
            "voting": consecutive_flags,
            "alert": is_alert
        })

    # Tính Waveform Peaks (1500 điểm)
    num_pts = 1500
    chunk_len = max(1, total_samples // num_pts)
    wf_peaks = []
    for i in range(num_pts):
        seg = audio_stream[i * chunk_len : (i + 1) * chunk_len]
        wf_peaks.append(float(np.max(np.abs(seg))) if len(seg) > 0 else 0.0)

    # Tính Librosa Mel-Spectrogram độ nét cao
    S = librosa.feature.melspectrogram(
        y=audio_stream, sr=TARGET_SR,
        n_fft=1024, hop_length=256, win_length=1024, window='hann',
        n_mels=128, fmin=100.0, fmax=5000.0, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=65.0)
    S_norm = np.clip((S_db + 65.0) / 65.0, 0.0, 1.0)
    colormap = cm.get_cmap('magma')
    spec_rgba = (colormap(S_norm) * 255).astype(np.uint8)

    # Xuất audio WAV 16-bit chuẩn sang base64 để browser phát trực tiếp
    mem_wav = io.BytesIO()
    sf.write(mem_wav, audio_stream, TARGET_SR, subtype='PCM_16', format='WAV')
    audio_b64 = base64.b64encode(mem_wav.getvalue()).decode("utf-8")

# ==========================================
# 4. TỔNG HỢP KẾT QUẢ & NÚT DOWNLOAD
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Thời lượng âm thanh", f"{duration:.2f}s")
col2.metric("Tổng số khung quét", f"{frame_idx} frames")
col3.metric("Số lần báo động cưa máy", f"{total_alerts} lần", delta="🚨 Cảnh báo" if total_alerts > 0 else None)
col4.metric("Tình trạng chung", "PHÁT HIỆN CƯA" if total_alerts > 0 else "AN TOÀN / NỀN")

# Xuất CSV và Audacity Labels
csv_buf = io.StringIO()
csv_writer = csv.DictWriter(csv_buf, fieldnames=['frame_idx', 't', 'rms', 'cf', 'sub_b', 'artifact', 'score', 'voting', 'alert'])
csv_writer.writeheader()
csv_writer.writerows(logs)

txt_buf = io.StringIO()
for item in logs:
    t_s = item['t']
    t_e = round(t_s + (WINDOW_SIZE / TARGET_SR), 4)
    if item['alert'] == 1:
        txt_buf.write(f"{t_s:.4f}\t{t_e:.4f}\t[ALERT] Chainsaw p={item['score']:.2f}\n")
    elif item['score'] >= conf_thresh:
        txt_buf.write(f"{t_s:.4f}\t{t_e:.4f}\tChainsaw p={item['score']:.2f}\n")
    elif item['artifact'] == 1:
        txt_buf.write(f"{t_s:.4f}\t{t_e:.4f}\tArtifact (Tap/Bump) CF={item['cf']:.1f}\n")

dl_col1, dl_col2 = st.columns(2)
dl_col1.download_button("📥 Tải File Log CSV", csv_buf.getvalue(), file_name="inference_logs.csv", mime="text/csv")
dl_col2.download_button("📥 Tải File Nhãn Audacity (.txt)", txt_buf.getvalue(), file_name="audacity_labels.txt", mime="text/plain")

# ==========================================
# 5. GIAO DIỆN TƯƠNG TÁC 60 FPS HTML5 + CANVAS
# ==========================================
component_payload = {
    "audio_b64": f"data:audio/wav;base64,{audio_b64}",
    "duration": duration,
    "waveform": wf_peaks,
    "spec_h": spec_rgba.shape[0],
    "spec_w": spec_rgba.shape[1],
    "spec_rgba": spec_rgba.flatten().tolist(),
    "logs": logs,
    "conf_thresh": conf_thresh,
    "voting_thresh": voting_thresh
}

html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #0e1117;
            color: #f0f2f6;
            margin: 0;
            padding: 10px;
        }}
        .player-bar {{
            background: #1e2530;
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        button {{
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{ background-color: #e03838; }}
        .time-box {{
            font-family: monospace;
            font-size: 15px;
            color: #58a6ff;
            min-width: 140px;
        }}
        .metric-badge {{
            background: #262c38;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 13px;
        }}
        .charts-wrapper {{
            position: relative;
            cursor: pointer;
        }}
        .chart-container {{
            position: relative;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 12px;
        }}
        .chart-title {{
            position: absolute;
            top: 6px;
            left: 10px;
            font-size: 12px;
            font-weight: bold;
            color: #8b949e;
            background: rgba(22, 27, 34, 0.85);
            padding: 2px 8px;
            border-radius: 4px;
            pointer-events: none;
            z-index: 10;
        }}
        canvas {{
            display: block;
            width: 100%;
        }}
        #globalPlayhead {{
            position: absolute;
            top: 0;
            bottom: 0;
            left: 0;
            width: 2px;
            background-color: #00ffcc;
            box-shadow: 0 0 10px #00ffcc;
            pointer-events: none;
            z-index: 50;
        }}
        #hoverLine {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 1px;
            border-left: 1px dashed #e3b341;
            pointer-events: none;
            display: none;
            z-index: 40;
        }}
        #hoverDot {{
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #ffff00;
            border: 2px solid #ffffff;
            box-shadow: 0 0 8px #ffff00;
            transform: translate(-50%, -50%);
            pointer-events: none;
            display: none;
            z-index: 45;
        }}
        #chartTooltip {{
            position: absolute;
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid #58a6ff;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
            line-height: 1.5;
            color: #c9d1d9;
            pointer-events: none;
            display: none;
            z-index: 60;
            box-shadow: 0 4px 14px rgba(0,0,0,0.6);
            backdrop-filter: blur(4px);
            min-width: 180px;
        }}
        #chartTooltip .tt-title {{
            font-weight: bold;
            color: #58a6ff;
            margin-bottom: 4px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 2px;
        }}
        #chartTooltip .tt-row {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }}
    </style>
</head>
<body>

    <audio id="audio" src="{component_payload['audio_b64']}"></audio>

    <div class="player-bar">
        <button id="btnPlay">▶ PLAY</button>
        <div class="time-box" id="timeBox">00:00.0 / 00:00.0</div>
        <div class="metric-badge">Xác suất: <strong id="scoreBadge" style="color: #ff7b72;">--%</strong></div>
        <div class="metric-badge">Voting: <strong id="voteBadge" style="color: #7ee787;">--/{component_payload['voting_thresh']}</strong></div>
        <div class="metric-badge">Trạng thái: <strong id="statusBadge" style="color: #d2a8ff;">Sẵn sàng</strong></div>
    </div>

    <div class="charts-wrapper" id="chartsWrapper">
        <div id="globalPlayhead"></div>

        <!-- 1. Waveform -->
        <div class="chart-container">
            <div class="chart-title">1. DẠNG SÓNG ÂM THANH (RAW WAVEFORM - 16 kHz)</div>
            <canvas id="cvWave" height="85"></canvas>
        </div>

        <!-- 2. Spectrogram -->
        <div class="chart-container">
            <div class="chart-title">2. ÂM PHỔ LIBROSA MEL-SPECTROGRAM ĐỘ NÉT CAO (100 - 5000 Hz)</div>
            <canvas id="cvSpec" height="175"></canvas>
        </div>

        <!-- 3. Confidence & Tooltip -->
        <div class="chart-container" id="containerConf">
            <div class="chart-title">3. ĐIỂM XÁC SUẤT CHAINSAW & BỘ LỌC BIỂU QUYẾT (VOTING)</div>
            <div id="hoverLine"></div>
            <div id="hoverDot"></div>
            <div id="chartTooltip"></div>
            <canvas id="cvConf" height="125"></canvas>
        </div>
    </div>

    <script>
        const data = {json.dumps(component_payload)};
        const audio = document.getElementById('audio');
        const btnPlay = document.getElementById('btnPlay');
        const timeBox = document.getElementById('timeBox');
        const scoreBadge = document.getElementById('scoreBadge');
        const voteBadge = document.getElementById('voteBadge');
        const statusBadge = document.getElementById('statusBadge');
        const playhead = document.getElementById('globalPlayhead');
        const chartsWrapper = document.getElementById('chartsWrapper');

        const containerConf = document.getElementById('containerConf');
        const hoverLine = document.getElementById('hoverLine');
        const hoverDot = document.getElementById('hoverDot');
        const chartTooltip = document.getElementById('chartTooltip');

        const cvWave = document.getElementById('cvWave');
        const cvSpec = document.getElementById('cvSpec');
        const cvConf = document.getElementById('cvConf');

        const ctxWave = cvWave.getContext('2d');
        const ctxSpec = cvSpec.getContext('2d');
        const ctxConf = cvConf.getContext('2d');

        function resizeCanvases() {{
            const w = chartsWrapper.clientWidth;
            cvWave.width = w;
            cvSpec.width = w;
            cvConf.width = w;
            renderAllCharts();
        }}
        window.addEventListener('resize', resizeCanvases);

        function drawWaveform() {{
            const w = cvWave.width;
            const h = cvWave.height;
            ctxWave.fillStyle = "#161b22";
            ctxWave.fillRect(0, 0, w, h);

            ctxWave.strokeStyle = "#388bfd";
            ctxWave.lineWidth = 1;
            ctxWave.beginPath();

            const peaks = data.waveform;
            const mid = h / 2;
            for (let i = 0; i < peaks.length; i++) {{
                const x = (i / peaks.length) * w;
                const amp = peaks[i] * (h / 2) * 0.9;
                ctxWave.moveTo(x, mid - amp);
                ctxWave.lineTo(x, mid + amp);
            }}
            ctxWave.stroke();
        }}

        function drawSpectrogram() {{
            const w = cvSpec.width;
            const h = cvSpec.height;
            
            const offCanvas = document.createElement('canvas');
            offCanvas.width = data.spec_w;
            offCanvas.height = data.spec_h;
            const offCtx = offCanvas.getContext('2d');
            const imgData = offCtx.createImageData(data.spec_w, data.spec_h);

            const rawBytes = data.spec_rgba;
            for (let r = 0; r < data.spec_h; r++) {{
                const srcRow = data.spec_h - 1 - r;
                for (let c = 0; c < data.spec_w; c++) {{
                    const srcIdx = (srcRow * data.spec_w + c) * 4;
                    const dstIdx = (r * data.spec_w + c) * 4;
                    imgData.data[dstIdx + 0] = rawBytes[srcIdx + 0];
                    imgData.data[dstIdx + 1] = rawBytes[srcIdx + 1];
                    imgData.data[dstIdx + 2] = rawBytes[srcIdx + 2];
                    imgData.data[dstIdx + 3] = 255;
                }}
            }}
            offCtx.putImageData(imgData, 0, 0);

            ctxSpec.imageSmoothingEnabled = true;
            ctxSpec.imageSmoothingQuality = 'high';
            ctxSpec.drawImage(offCanvas, 0, 0, w, h);
        }}

        function drawConfidence() {{
            const w = cvConf.width;
            const h = cvConf.height;
            ctxConf.fillStyle = "#161b22";
            ctxConf.fillRect(0, 0, w, h);

            if (!data.logs || data.logs.length === 0) return;

            // Ngưỡng phân loại đã cấu hình
            const threshY = h - (data.conf_thresh * (h * 0.82) + h * 0.08);
            ctxConf.strokeStyle = "#484f58";
            ctxConf.setLineDash([4, 4]);
            ctxConf.beginPath();
            ctxConf.moveTo(0, threshY);
            ctxConf.lineTo(w, threshY);
            ctxConf.stroke();
            ctxConf.setLineDash([]);

            // Đường xác suất p
            ctxConf.strokeStyle = "#ff7b72";
            ctxConf.lineWidth = 2;
            ctxConf.beginPath();
            data.logs.forEach((item, idx) => {{
                const x = (item.t / data.duration) * w;
                const y = h - (item.score * (h * 0.82) + h * 0.08);
                if (idx === 0) ctxConf.moveTo(x, y);
                else ctxConf.lineTo(x, y);
            }});
            ctxConf.stroke();

            // Điểm kích hoạt báo động
            data.logs.forEach((item) => {{
                if (item.alert === 1) {{
                    const x = (item.t / data.duration) * w;
                    const y = h - (item.score * (h * 0.82) + h * 0.08);
                    ctxConf.fillStyle = "#f85149";
                    ctxConf.beginPath();
                    ctxConf.arc(x, y, 4.5, 0, Math.PI * 2);
                    ctxConf.fill();
                }}
            }});
        }}

        function renderAllCharts() {{
            drawWaveform();
            drawSpectrogram();
            drawConfidence();
        }}

        // Tooltip khi rà chuột
        containerConf.addEventListener('mousemove', (e) => {{
            if (!data.logs || data.logs.length === 0) return;

            const rect = containerConf.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const w = containerConf.clientWidth;
            const h = containerConf.clientHeight;

            const hoverTime = (mouseX / w) * data.duration;
            let closestLog = data.logs[0];
            let minDiff = Math.abs(data.logs[0].t - hoverTime);
            for (let i = 1; i < data.logs.length; i++) {{
                const diff = Math.abs(data.logs[i].t - hoverTime);
                if (diff < minDiff) {{
                    minDiff = diff;
                    closestLog = data.logs[i];
                }}
            }}

            const logX = (closestLog.t / data.duration) * w;
            const logY = h - (closestLog.score * (h * 0.82) + h * 0.08);

            hoverLine.style.display = 'block';
            hoverLine.style.left = `${{logX}}px`;

            hoverDot.style.display = 'block';
            hoverDot.style.left = `${{logX}}px`;
            hoverDot.style.top = `${{logY}}px`;

            let statusText = "An toàn (Nền)";
            let statusColor = "#7ee787";

            if (closestLog.alert === 1) {{
                statusText = "🚨 CẢNH BÁO LORAWAN!";
                statusColor = "#ff7b72";
            }} else if (closestLog.artifact === 1) {{
                statusText = "🛑 XUNG GÕ MIC";
                statusColor = "#d2a8ff";
            }} else if (closestLog.score >= data.conf_thresh) {{
                statusText = `⚠️ Nghi vấn (≥ ${{data.conf_thresh*100}}%)`;
                statusColor = "#e3b341";
            }}

            chartTooltip.innerHTML = `
                <div class="tt-title">⏱️ Mốc: ${{closestLog.t.toFixed(2)}}s (${{formatTime(closestLog.t)}})</div>
                <div class="tt-row"><span>Xác suất (p):</span><strong style="color: ${{closestLog.score >= data.conf_thresh ? '#ff7b72' : '#58a6ff'}}">${{(closestLog.score * 100).toFixed(1)}}%</strong></div>
                <div class="tt-row"><span>Chuỗi Voting:</span><strong>${{closestLog.voting}}/${{data.voting_thresh}}</strong></div>
                <div class="tt-row"><span>Crest Factor:</span><strong>${{closestLog.cf.toFixed(1)}}</strong></div>
                <div class="tt-row"><span>Sub-bass (<200Hz):</span><strong>${{(closestLog.sub_b * 100).toFixed(0)}}%</strong></div>
                <div class="tt-row" style="margin-top: 4px; padding-top: 3px; border-top: 1px dashed #30363d;">
                    <span>Phán quyết:</span><strong style="color: ${{statusColor}}">${{statusText}}</strong>
                </div>
            `;

            chartTooltip.style.display = 'block';
            const tooltipWidth = 190;
            if (logX + tooltipWidth + 15 > w) {{
                chartTooltip.style.left = `${{logX - tooltipWidth - 15}}px`;
            }} else {{
                chartTooltip.style.left = `${{logX + 15}}px`;
            }}
            chartTooltip.style.top = `15px`;
        }});

        containerConf.addEventListener('mouseleave', () => {{
            hoverLine.style.display = 'none';
            hoverDot.style.display = 'none';
            chartTooltip.style.display = 'none';
        }});

        // Quản lý Play/Pause
        btnPlay.addEventListener('click', () => {{
            if (audio.paused) {{
                audio.play();
                btnPlay.textContent = "⏸ PAUSE";
            }} else {{
                audio.pause();
                btnPlay.textContent = "▶ PLAY";
            }}
        }});

        function formatTime(sec) {{
            const m = Math.floor(sec / 60);
            const s = (sec % 60).toFixed(1);
            return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
        }}

        function updatePlayhead() {{
            const cur = audio.currentTime;
            const dur = data.duration;
            const progress = Math.min(1, cur / dur);
            const pixelX = progress * chartsWrapper.clientWidth;

            playhead.style.transform = `translateX(${{pixelX}}px)`;
            timeBox.textContent = `${{formatTime(cur)}} / ${{formatTime(dur)}}`;

            if (data.logs && data.logs.length > 0) {{
                let activeLog = data.logs[0];
                for (let i = 0; i < data.logs.length; i++) {{
                    if (data.logs[i].t <= cur) activeLog = data.logs[i];
                    else break;
                }}
                scoreBadge.textContent = (activeLog.score * 100).toFixed(1) + "%";
                voteBadge.textContent = activeLog.voting + "/" + data.voting_thresh;
                
                if (activeLog.alert === 1) {{
                    statusBadge.textContent = "🚨 BÁO ĐỘNG CƯA MÁY";
                    statusBadge.style.color = "#ff7b72";
                }} else if (activeLog.artifact === 1) {{
                    statusBadge.textContent = "🛑 XUNG GÕ MIC";
                    statusBadge.style.color = "#d2a8ff";
                }} else if (activeLog.score >= data.conf_thresh) {{
                    statusBadge.textContent = "Nghi vấn (Vượt ngưỡng)";
                    statusBadge.style.color = "#e3b341";
                }} else {{
                    statusBadge.textContent = "An toàn (Nền)";
                    statusBadge.style.color = "#7ee787";
                }}
            }}

            if (!audio.paused) {{
                requestAnimationFrame(updatePlayhead);
            }}
        }}

        audio.addEventListener('play', () => {{
            requestAnimationFrame(updatePlayhead);
        }});
        audio.addEventListener('pause', () => {{
            btnPlay.textContent = "▶ PLAY";
        }});
        audio.addEventListener('ended', () => {{
            btnPlay.textContent = "▶ PLAY";
        }});

        // Tua nhanh khi bấm vào đồ thị
        chartsWrapper.addEventListener('click', (e) => {{
            const rect = chartsWrapper.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const seekPct = Math.max(0, Math.min(1, clickX / rect.width));
            audio.currentTime = seekPct * data.duration;
            updatePlayhead();
        }});

        setTimeout(resizeCanvases, 100);
    </script>
</body>
</html>
"""

components.html(html_code, height=640)