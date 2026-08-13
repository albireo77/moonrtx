"""
VideoMixin: time-lapse MP4 export and the text burned into its frames
(local time and an optional caption) for MoonRenderer.
"""

import unicodedata
import cv2
import numpy as np
from typing import Optional
from datetime import datetime


class VideoMixin:
    """Mixin providing time-lapse video export methods for MoonRenderer."""

    # Burned-in text for exported videos: the local time of each frame in the
    # bottom-left corner and an optional user caption in the top-left corner.
    # The Tk status bar is not part of the ray-traced image, so both are
    # composited into the frame by the Overlay postprocessing stage: an RGBA
    # texture blended over the tone-mapped image, which the NVENC encoder then
    # captures along with it. Verified on PlotOptiX 0.19.2: blending is exact
    # alpha compositing (an opaque black patch renders 0, a 50% black patch
    # over 46 renders 23), and every encoded frame carries the overlay that was
    # set before its own accumulation cycle started - marker sequence
    # 30/70/110/150/190/230 came back in order with no off-by-one.
    # Postprocessing stages cannot be removed once added, so "nothing burned
    # in" means an all-transparent texture.
    VIDEO_OVERLAY_TEXTURE = "frame_overlay"
    VIDEO_TIME_TEXT_FRACTION = 0.015    # text height as a fraction of frame height
    VIDEO_TIME_MARGIN_FRACTION = 0.015
    VIDEO_TIME_BOX_ALPHA = 150          # backing box, keeps text readable over terrain
    VIDEO_CAPTION_MAX_CHARS = 100
    # Corners the time and the caption can be placed in; the two must differ
    VIDEO_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
    VIDEO_TIME_CORNER = "bottom-left"
    VIDEO_CAPTION_CORNER = "top-left"

    def _init_video_export(self):
        """Reset export state; called from MoonRenderer.__init__."""
        self._video_export = None
        self._video_encoder_cfg = None
        self._video_overlay_buf = None
        self._video_overlay_ready = False

    # ---- text burned into the frames ----

    def _video_time_text(self, dt_local: datetime) -> str:
        """
        Local time as burned into exported video frames: same format as the
        status bar time panel, without its time-step suffix.
        """
        offset = dt_local.strftime('%z')
        offset_fmt = f"{offset[:3]}:{offset[3:]}" if offset else ""
        return f"{dt_local.strftime('%Y-%m-%d %H:%M:%S')}{offset_fmt}"

    @staticmethod
    def _video_ascii(text: str) -> str:
        """
        Fold text into the ASCII range cv2.putText can draw: the Hershey fonts
        it uses have no glyphs beyond it, so anything else would come out as
        '?'. Accented Latin letters lose their marks ("Kraków" stays readable
        as "Krakow"), letters that do not decompose are mapped explicitly, and
        whatever is left really does become '?'.
        """
        specials = {'ł': 'l', 'Ł': 'L', 'ø': 'o', 'Ø': 'O', 'đ': 'd', 'Đ': 'D',
                    'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE', 'ß': 'ss',
                    '–': '-', '—': '-', '’': "'", '‘': "'", '“': '"', '”': '"',
                    '°': ' deg'}
        text = ''.join(specials.get(c, c) for c in text)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        return ''.join(c if 32 <= ord(c) < 127 else '?' for c in text)

    def _draw_video_label(self, buf, text: str, corner: str):
        """
        Draw one line of white text on a translucent dark box in one corner of
        the overlay (see VIDEO_CORNERS). The box keeps the text readable over
        both the black sky and sunlit terrain, and the text is sized to the
        render resolution so it looks the same at any window size.

        In the right-hand corners the box is anchored to the right margin and
        grows leftwards, so the whole label stays inside the frame instead of
        running off the edge.
        """
        h, w = buf.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        px = max(10, int(round(h * self.VIDEO_TIME_TEXT_FRACTION)))
        thickness = max(1, px // 9)
        scale = cv2.getFontScaleFromHeight(font, px, thickness)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = max(6, int(round(h * self.VIDEO_TIME_MARGIN_FRACTION)))
        pad = max(4, px // 3)
        box_w = tw + 2 * pad
        box_h = th + baseline + 2 * pad
        top = corner.startswith("top")
        right = corner.endswith("right")
        y0 = margin if top else max(0, h - margin - box_h)
        y1 = min(h - 1, y0 + box_h)
        x0 = max(0, w - margin - box_w) if right else margin
        x1 = min(w - 1, x0 + box_w)
        cv2.rectangle(buf, (x0, y0), (x1, y1),
                      (0, 0, 0, self.VIDEO_TIME_BOX_ALPHA), cv2.FILLED)
        cv2.putText(buf, text, (x0 + pad, y1 - pad - baseline),
                    font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)

    def _set_video_overlay(self, dt_local: Optional[datetime] = None, caption: str = "",
                           time_corner: str = VIDEO_TIME_CORNER,
                           caption_corner: str = VIDEO_CAPTION_CORNER):
        """
        Compose the video frame overlay: the local time of dt_local and the
        caption, each in its own corner. Either can be omitted, and omitting
        both clears the overlay to fully transparent - which is how it is
        switched off, the postprocessing stage itself being impossible to
        remove once added.

        Must be called before the accumulation cycle of the frame it belongs
        to starts: the encoder captures the overlay in place at that moment
        (see the VIDEO_OVERLAY_TEXTURE comment block).
        """
        if self.rt is None:
            return

        h, w = self.rt._height, self.rt._width
        buf = self._video_overlay_buf
        if buf is None or buf.shape[0] != h or buf.shape[1] != w:
            buf = np.zeros((h, w, 4), dtype=np.uint8)
            self._video_overlay_buf = buf
        else:
            buf.fill(0)

        if dt_local is not None:
            self._draw_video_label(buf, self._video_time_text(dt_local), time_corner)
        if caption:
            self._draw_video_label(buf, caption, caption_corner)

        # Nearest filtering keeps the text crisp at the 1:1 texture/frame match
        self.rt.set_texture_2d(self.VIDEO_OVERLAY_TEXTURE, buf,
                               filter_mode="Nearest", refresh=False)
        if not self._video_overlay_ready:
            # Added after init_renderer's Gamma stage, so the text is
            # composited onto the tone-mapped image and keeps exactly the
            # colors set above
            self.rt.add_postproc("Overlay")
            self._video_overlay_ready = True

    # ---- export control ----

    def start_video_export(self, filename: str, n_frames: int, step_minutes: int,
                           fps: int, bitrate: float, on_progress, on_done,
                           burn_time: bool = True, caption: str = "",
                           time_corner: str = VIDEO_TIME_CORNER,
                           caption_corner: str = VIDEO_CAPTION_CORNER) -> Optional[str]:
        """
        Start a time-lapse export: from the current observation time, advance
        by step_minutes per video frame, letting every frame converge to the
        full accumulation quality before it is encoded. Verified on PlotOptiX
        0.19.2: the NVENC encoder grabs exactly one frame per completed
        accumulation cycle and stops itself after n_frames, so the export is
        driven from the accum-done callback with no frame drops/duplicates.

        Note: fps and bitrate are fixed by the first export of the session.
        PlotOptiX supports a single encoder_create per raytracer instance and
        silently ignores re-creation with new settings (verified).

        Parameters
        ----------
        filename : str
            Output MP4 file path
        n_frames : int
            Number of video frames to render
        step_minutes : int
            Simulated time step between frames (negative runs backwards)
        fps : int
            Playback frame rate (first export of the session only)
        bitrate : float
            H.264 bitrate in Mbit/s (first export of the session only)
        on_progress : callable
            on_progress(frame, total, dt_local), called on the Tk main thread
            after each frame
        on_done : callable
            on_done(error_or_none), called on the Tk main thread when the
            export finished, failed, or was cancelled
        burn_time : bool
            Draw each frame's local time into the video, bottom-left
        caption : str
            Optional free text to draw in every frame. Blank (or
            whitespace-only) means no caption; longer text is cut to
            VIDEO_CAPTION_MAX_CHARS and folded to ASCII (see _video_ascii).
        time_corner, caption_corner : str
            Corners to place them in, one of VIDEO_CORNERS. They must differ,
            or the two labels would be drawn on top of each other.

        Returns
        -------
        str or None
            Error message, or None when the export has started.
        """
        if self.rt is None:
            return "Renderer not running"
        if self._video_export is not None:
            return "An export is already running"

        # Auto-advance ticks would inject extra time jumps mid-export, so it is
        # switched off for the duration and handed back afterwards
        resume_auto_advance = bool(self._auto_advance_var and self._auto_advance_var.get())
        if resume_auto_advance:
            self._auto_advance_var.set(False)
            self._on_auto_advance_toggle()

        # Make sure converged accumulation is active: a pending interactive
        # preview would make the encoder capture single-pass noisy frames
        if self._preview_restore_id is not None:
            self.rt._root.after_cancel(self._preview_restore_id)
            self._preview_restore_id = None
        if self._preview_active:
            self._preview_active = False
            self.rt.set_param(max_accumulation_frames=self.ACCUMULATION_FRAMES)

        first_create = self._video_encoder_cfg is None
        if first_create:
            try:
                self.rt.encoder_create(fps=fps, bitrate=bitrate)
            except Exception as e:
                return f"Video encoder not available: {e}"

        self._video_export = {
            "n": n_frames,
            "step": step_minutes,
            "frame": 0,
            "cancel": False,
            "on_progress": on_progress,
            "on_done": on_done,
            "burn_time": burn_time,
            "caption": self._video_ascii(caption.strip())[:self.VIDEO_CAPTION_MAX_CHARS],
            "time_corner": time_corner,
            "caption_corner": caption_corner,
            "resume_auto_advance": resume_auto_advance,
        }

        try:
            self.rt.encoder_start(filename, n_frames)
        except Exception as e:
            self._video_export = None
            return f"Video encoder not started: {e}"

        # Definitive success check. PlotOptiX reports encoder failures only in
        # the log (default _raise_on_error=False), so without this check an
        # export with e.g. missing FFmpeg DLLs would render every frame with
        # no encoder attached and silently produce no file. A successfully
        # started encoder is always open (verified on Windows for both the
        # missing-FFmpeg and NVENC-error cases).
        if not self.rt.encoder_is_open():
            self._video_export = None
            return ("Video encoder failed to start - no frames were rendered.\n"
                    "Check that FFmpeg (shared build) DLLs are in PATH; details "
                    "are in the console output.")
        if first_create:
            self._video_encoder_cfg = (fps, bitrate)

        self.rt.set_accum_done_cb(self._video_export_accum_done)
        # Re-render the current time: it becomes the first video frame, so its
        # burned-in text has to be in place before the cycle starts
        st = self._video_export
        if burn_time or st["caption"]:
            with self.rt._padlock:
                self._set_video_overlay(self.dt_local if burn_time else None, st["caption"],
                                        time_corner, caption_corner)
        self.rt.refresh_scene()
        return None

    def cancel_video_export(self):
        """Request export cancellation; it stops after the current frame."""
        if self._video_export is not None:
            self._video_export["cancel"] = True

    def _video_export_accum_done(self, rt):
        """
        Accum-done callback driving the video export. Runs on the raytracing
        thread with the render padlock held (an RLock, so the padlock use
        inside update_view is re-entrant); GUI work is posted to the Tk
        main thread.
        """
        st = self._video_export
        if st is None:
            return
        st["frame"] += 1
        error = None

        if not st["cancel"] and st["frame"] < st["n"]:
            if not rt.encoder_is_open():
                # The encoder auto-closes only at the frame limit; closed
                # earlier means an encoding failure - stop instead of
                # rendering the remaining frames into the void
                error = "Video encoder closed unexpectedly (see console output)."
            else:
                # Advance to the next frame's time; update_view refreshes the
                # scene, which starts the next accumulation cycle. The overlay
                # is drawn first, so it is already in place when that cycle -
                # the frame it labels - begins.
                # Minutes of real time, not of wall clock (see shifted_time),
                # then put on the observer's clock: shifted_time answers in UTC,
                # and this value is burned into the frame as well as rendered,
                # so it has to read the same as the status bar
                next_dt = self.in_observer_clock(self.shifted_time(st["step"]))
                try:
                    with self.rt._padlock:
                        if st["burn_time"] or st["caption"]:
                            self._set_video_overlay(
                                next_dt if st["burn_time"] else None, st["caption"],
                                st["time_corner"], st["caption_corner"])
                        self.update_view(next_dt)
                except Exception as e:
                    # E.g. the date left the supported ephemeris range: end the
                    # export with the partial file. update_view rejects such a
                    # date before changing anything, so the renderer stays on the
                    # last valid time and the frame carrying the rejected
                    # timestamp is never encoded.
                    error = str(e)

        if st["cancel"] or st["frame"] >= st["n"] or error is not None:
            self._video_export = None
            self.rt.set_accum_done_cb(None)

            def finish():
                """
                Release everything the export took over, so the interactive
                view behaves exactly as it did before it started. Runs on the
                Tk main thread half a second later, giving NVENC time to flush
                the last frame.
                """
                # Should another export have started meanwhile, it now owns the
                # encoder and the overlay: only report this one's outcome
                if self._video_export is not None:
                    st["on_done"](error)
                    return

                # When the frame limit was reached the encoder closed itself;
                # after a cancel or an error it is still open and stops here
                if self.rt is not None and self.rt.encoder_is_open():
                    self.rt.encoder_stop()

                if self.rt is not None and (st["burn_time"] or st["caption"]):
                    # Take the burned-in text back out of the live view. The
                    # postprocessing stage cannot be removed once added, so an
                    # all-transparent texture is what makes it a no-op; the
                    # GPU keeps its own copy, so the host-side buffer (a full
                    # RGBA frame, ~8 MB at 1920x1040) can go.
                    with self.rt._padlock:
                        self._set_video_overlay()
                    self._video_overlay_buf = None
                    self.rt.refresh_scene()

                # Hand auto-advance back if the export took it away
                if st["resume_auto_advance"] and self._auto_advance_var is not None:
                    self._auto_advance_var.set(True)
                    self._on_auto_advance_toggle()

                self._update_all_status_panels()
                st["on_done"](error)

            self.rt._root.after(500, finish)
        else:
            self.rt._root.after(0, self._video_export_progress,
                                st["on_progress"], st["frame"], st["n"], self.dt_local)

    def _video_export_progress(self, cb, frame: int, total: int, dt_local: datetime):
        """Per-frame GUI update (Tk main thread): app status plus dialog callback."""
        self._update_status_time()
        self._update_info_moon()
        cb(frame, total, dt_local)
