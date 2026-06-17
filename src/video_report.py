def frames_to_ranges(frames, fps, merge_gap_frames=0, min_range_frames=1):
    if not frames:
        return []

    ranges = []
    start = frames[0]
    previous = frames[0]

    for frame in frames[1:]:
        if frame <= previous + merge_gap_frames + 1:
            previous = frame
            continue

        frame_count = previous - start + 1
        if frame_count >= min_range_frames:
            ranges.append(make_range(start, previous, fps))

        start = frame
        previous = frame

    frame_count = previous - start + 1
    if frame_count >= min_range_frames:
        ranges.append(make_range(start, previous, fps))

    return ranges


def make_range(start_frame, end_frame, fps):
    frame_count = end_frame - start_frame + 1

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_seconds": round(start_frame / fps, 2),
        "end_time_seconds": round(end_frame / fps, 2),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 2),
    }


def make_review_event(event_type, priority, reason, suggested_action, item):
    return {
        "type": event_type,
        "priority": priority,
        "start_frame": item["start_frame"],
        "end_frame": item["end_frame"],
        "start_time_seconds": item["start_time_seconds"],
        "end_time_seconds": item["end_time_seconds"],
        "frame_count": item["frame_count"],
        "duration_seconds": item["duration_seconds"],
        "reason": reason,
        "suggested_action": suggested_action,
    }


def build_review_section(report, fps):
    no_face_ranges = frames_to_ranges(
        report["frames_with_no_face_detection"],
        fps,
    )

    no_head_ranges = frames_to_ranges(
        report["frames_with_no_head_detection"],
        fps,
    )

    no_face_but_head_ranges = frames_to_ranges(
        report["frames_with_no_face_but_head_detected"],
        fps,
    )

    no_face_or_head_ranges = frames_to_ranges(
        report["frames_with_no_face_or_head_detection"],
        fps,
    )

    sustained_held_ranges = frames_to_ranges(
        report["frames_with_held_face_or_head_boxes"],
        fps,
        merge_gap_frames=5,
        min_range_frames=10,
    )

    suggested_ranges = []

    for item in no_face_or_head_ranges:
        suggested_ranges.append(make_review_event(
            "no_face_or_head_detection",
            "high",
            "No current face or head detection was available, so automatic privacy coverage may be missing.",
            "Review this range carefully in the blurred and debug videos. Add manual extra blur if a patient or staff face/head is visible.",
            item,
        ))

    for item in sustained_held_ranges:
        suggested_ranges.append(make_review_event(
            "sustained_held_face_or_head_boxes",
            "medium",
            "Automatic blur used held face/head boxes during sustained detector dropout.",
            "Check whether the held blur box still covers the correct face/head area throughout this range.",
            item,
        ))

    warnings = []

    if no_face_or_head_ranges:
        warnings.append(
            "Some ranges had no current face or head detection. Review these carefully."
        )

    if sustained_held_ranges:
        warnings.append(
            "Some ranges used tracker-held blur boxes. Review these ranges to confirm blur coverage."
        )

    return {
        "review": {
            "summary": build_review_summary(suggested_ranges, warnings),
            "warnings": warnings,
            "suggested_ranges": suggested_ranges,
            "debug_ranges": {
                "no_face_detection": no_face_ranges,
                "no_head_detection": no_head_ranges,
                "no_face_but_head_detected": no_face_but_head_ranges,
                "held_face_or_head_boxes": sustained_held_ranges,
                "no_face_or_head_detection": no_face_or_head_ranges,
            },
        }
    }


def build_review_summary(suggested_ranges, warnings):
    frame_count = 0
    duration_seconds = 0

    for item in suggested_ranges:
        frame_count += item["frame_count"]
        duration_seconds += item["duration_seconds"]

    return {
        "suggested_review_range_count": len(suggested_ranges),
        "suggested_review_frame_count": frame_count,
        "suggested_review_duration_seconds": round(duration_seconds, 2),
        "warning_count": len(warnings),
    }