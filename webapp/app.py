"""
RNNoise Web App - Flask Backend
Uses the compiled C tool for fast audio denoising
"""
from flask import Flask, request, send_file, render_template, jsonify
import subprocess
import os
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
RNNOISE_DEMO = os.path.join(BASE_DIR, "..", "examples", "rnnoise_demo")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def convert_to_raw_pcm(input_path, output_raw):
    """
    Convert ANY input audio into headerless raw 16-bit PCM, mono, 48kHz.
    Uses the soxr resampler (higher quality than ffmpeg's default swresample)
    to minimize aliasing/ringing artifacts introduced during sample-rate
    conversion - important since the model was trained on clean PCM.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "aresample=resampler=soxr:precision=28:dither_method=triangular",
        "-ar", "48000",
        "-ac", "1",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        output_raw
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback if this ffmpeg build lacks libsoxr support
        if "soxr" in result.stderr.lower() or "Unknown resampler" in result.stderr:
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", "48000",
                "-ac", "1",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                output_raw
            ]
            result = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg error (input->raw, fallback): {result.stderr}")
        else:
            raise Exception(f"FFmpeg error (input->raw): {result.stderr}")


def convert_raw_to_wav(input_raw, output_wav):
    """Wrap denoised raw PCM output into a playable WAV file"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le", "-ar", "48000", "-ac", "1",
        "-i", input_raw,
        "-c:a", "pcm_s16le",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error (raw->wav): {result.stderr}")


def denoise_audio(input_raw, output_raw):
    """Run the C RNNoise tool on headerless raw PCM, in and out."""
    if not os.path.exists(RNNOISE_DEMO):
        raise Exception(f"rnnoise_demo not found at {RNNOISE_DEMO}")
    cmd = [RNNOISE_DEMO, input_raw, output_raw]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"RNNoise error: {result.stderr}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/denoise", methods=["POST"])
def denoise():
    job_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_input")
    input_raw = os.path.join(UPLOAD_FOLDER, f"{job_id}_48k.raw")
    output_raw = os.path.join(PROCESSED_FOLDER, f"{job_id}_output.raw")
    output_wav = os.path.join(PROCESSED_FOLDER, f"{job_id}_denoised.wav")

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        file.save(input_path)

        convert_to_raw_pcm(input_path, input_raw)
        denoise_audio(input_raw, output_raw)
        convert_raw_to_wav(output_raw, output_wav)

        return jsonify({
            "success": True,
            "download_url": f"/download/{job_id}_denoised.wav",
            "message": "Audio denoised successfully!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for f in (input_path, input_raw, output_raw):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


@app.route("/download/<filename>")
def download(filename):
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    print("🎧 RNNoise Web App running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
