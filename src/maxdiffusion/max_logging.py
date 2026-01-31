"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Stub for logging utilities. Right now just meant to avoid raw prints"""

import os


def log(user_str):
  print(user_str, flush=True)


def _env_truthy(name: str) -> bool:
  value = os.environ.get(name, "").strip().lower()
  return value in {"1", "true", "yes", "y", "on"}


def timing_enabled() -> bool:
  # Default: disabled.
  return _env_truthy("F5_SHOW_TIMING") or _env_truthy("MAXDIFFUSION_SHOW_TIMING")


def log_timing(user_str):
  if timing_enabled():
    log(user_str)
