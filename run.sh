#!/usr/bin/env bash
cat > ~/.asoundrc <<'EOF'
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
EOF