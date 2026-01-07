# ----------------------------
# 🎯 SMART VOICE-BASED INTERRUPT DETECTION
# ----------------------------


def update_baseline(conn: WSConn, energy: int):
    """Update background noise baseline with improved adaptivity"""
    if not conn.currently_speaking:
        if energy < max(conn.baseline_energy * 2, 600):
            conn.background_samples.append(energy)
            if len(conn.background_samples) >= 20:
                recent_samples = list(conn.background_samples)[-20:]
                sorted_samples = sorted(recent_samples)
                weighted_median = sorted_samples[len(sorted_samples) // 2]
                conn.baseline_energy = (
                    conn.baseline_energy * 0.7) + (weighted_median * 0.3)


# ----------------------------
# âš¡ Interrupt Handler
# ----------------------------

async def handle_interrupt(call_sid: str):
    """Handle user interruption with complete cleanup"""
    conn = manager.get(call_sid)
    if not conn:
        return

    _logger.info("ðŸ›‘ INTERRUPT - Stopping playback and clearing buffers")

    conn.interrupt_requested = True

    cleared = 0
    while not conn.tts_queue.empty():
        try:
            conn.tts_queue.get_nowait()
            conn.tts_queue.task_done()
            cleared += 1
        except:
            break

    try:
        if conn.stream_sid:  # ✅ Validate stream_sid exists
            await conn.ws.send_json({
                "event": "clear",
                "streamSid": conn.stream_sid
            })
    except:
        pass

    old_buffer = conn.stt_transcript_buffer
    conn.stt_transcript_buffer = ""
    conn.stt_is_final = False
    conn.last_transcript = ""

    conn.currently_speaking = False
    conn.is_responding = False
    conn.speech_energy_buffer.clear()
    conn.speech_start_time = None
    conn.user_speech_detected = False
    conn.last_speech_time = 0
    conn.silence_start = None

    conn.last_interim_text = ""
    conn.last_interim_time = 0.0
    conn.last_interim_conf = 0.0

    _logger.info(
        "âœ… Interrupt handled:\n"
        "   Cleared TTS items: %d\n"
        "   Cleared STT buffer: '%s'\n"
        "   Ready for new input",
        cleared, old_buffer[:50] if old_buffer else "(empty)"
    )