#!/bin/bash
# Re-render the 8 animation clips. Needs node + playwright (npm i playwright; npx playwright install chromium) and ffmpeg.
#   ./make_videos.sh          all 8 clips
#   ./make_videos.sh 3 6      only clips 3 and 6
set -e
cd "$(dirname "$0")"
node render_frames.js "$(pwd)" "$@"
names=(1_system 2_sampling 3_mlp 4_kan 5_arrow_map 6_ode_solver 7_training 8_result)
mkdir -p videos
for k in ${@:-1 2 3 4 5 6 7 8}; do
  ffmpeg -y -loglevel error -framerate 25 -i frames_$k/f%05d.png -c:v libx264 -preset veryfast -crf 20 \
         -pix_fmt yuv420p -movflags +faststart videos/${names[$((k-1))]}.mp4
  rm -rf frames_$k
  echo "videos/${names[$((k-1))]}.mp4"
done
