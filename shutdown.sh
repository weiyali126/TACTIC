#!/bin/bash
# -*- coding: utf-8 -*-

# # 关闭 vllm
# ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}' | xargs -r kill

# 关闭 uvicorn backend
ps aux | grep "uvicorn tactic.app.server:app" | grep -v grep | awk '{print $2}' | xargs -r kill

# 关闭 frontend gradio
ps aux | grep "python -m tactic.app.frontend" | grep -v grep | awk '{print $2}' | xargs -r kill

echo "All servers stopped."
