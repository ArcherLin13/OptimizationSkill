# Text-heavy test case (729x126, app rect mask).
#
# Regenerate BMPs on PC:
#   python scripts/gen_text_case_bmp.py samples/text_case
#   # or without Python:
#   .\scripts\gen_text_case_bmp.ps1
#
# Push to phone:
#   .\scripts\push_case.ps1 -CaseDir samples\text_case
#
# Run on device (images -> ./out/):
#   cd /data/vendor/camera
#   ./seamless_clone_bench --case text
#
# Or use built-in generator on device:
#   ./seamless_clone_bench --case text --dump-case ./text_case
