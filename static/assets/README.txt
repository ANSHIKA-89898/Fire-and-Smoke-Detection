Place your alarm sound file here named:

    alarm.wav

This is the file the browser plays via the <audio> tag in index.html when
an alert fires. If it's missing, playback simply fails silently and the
browser console will show a harmless 404 — visual alerts still work.

You can also just copy the same file you place in the project-level
assets/ folder:
    cp assets/alarm.wav static/assets/alarm.wav   (macOS/Linux)
    copy assets\alarm.wav static\assets\alarm.wav (Windows)
