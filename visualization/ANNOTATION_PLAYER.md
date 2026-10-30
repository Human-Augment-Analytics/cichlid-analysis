# Full-video annotation player

Open annotation_player.html directly in a desktop browser, or serve the repository with `python3 -m http.server 8000` and visit http://localhost:8000/visualization/annotation_player.html . No model, Parquet, annotation CSV, Python packages, or clip extraction is required. A supported local video (such as H.264 MP4) is required.

1. Click the Video file chooser and select the original MP4. It is accessed locally through a browser object URL, not uploaded or copied into memory in full by this code.
2. Set the FPS to the source's verified constant frame rate. The supplied 0007 recording reports 30 FPS. Enter the team's agreed Video ID.
3. Use the native video controls or the timeline slider to scrub anywhere in the recording. Change speed for scanning. Large-file seeking depends on the source encoding/browser and may take a moment.
4. Pause and step one or six frames to inspect candidate boundaries. Jump directly to a frame number or elapsed HH:MM:SS.
5. Mark start and last included end frame. Select label/direction, enter turn count/location/notes, and click Add event. Uncertain bower location can remain blank. The user supplies all behavioral decisions.
6. Export CSV for the annotation sheet and JSON as a reloadable backup. Browser storage is convenience only; always export before ending a session. Use Edit to revise a saved row; Clear markers cancels an edit. Restoring a backup replaces the list after confirmation.

Frame numbers are **zero-based, computed as floor(video presentation time × configured FPS)**. The player uses requestVideoFrameCallback when available to read displayed-frame timestamps; seeked/timeupdate provide fallback updates. This is not independent verification of source decode order. For a constant-rate 30-FPS source starting at timestamp zero, it is the expected source frame mapping. Variable rate, nonzero time offsets, or mismatched FPS invalidate that assumption. Verify with the reviewer before treating these as exact ground-truth boundaries. Step buttons use time-based seeking; they are not a decoder-level previous-frame API. Current guidelines seek approximately 6–10-frame boundary precision.

Exported columns match the supplied annotation sheet. Start Frame and End Frame are inclusive; EndTime is the last included frame's time, not the exclusive-end convention in older timestamp CSVs. Do not feed this new CSV into the older model adapter without an explicit column/label/boundary adapter. Approximate-circling rows are manual labels, not automatically assigned training negatives here.

No pose overlay or automatic circling identification is implemented in this full-video mode. The existing short-clip viewer and training/hypothesis commands are unchanged. Record reviewed timeline coverage separately; reviewing candidates alone does not establish full-video annotation completeness.
