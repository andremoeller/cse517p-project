# Overview

Instructions for development.

## Development

Either open project root directory in VSCode and load devcontainer, or build via `make`:

- `make build-runtime` to build the minimal runtime image `evabyte:runtime`
- `make build-dev` to build the development image
- `run-runtime` to run an interactive shell in the minimal runtime image
- `run-dev` to run an interactive shell in the development image
- `shell-dev` to open an additional shell in the development image

## Notes

Various notes and thoughts that occurred that we may want to keep in mind.

1. According to Ed in [this discussion](https://edstem.org/us/courses/77432/discussion/6630668), timing starts when "docker build -t" _starts_, rather than when it ends. This means optimizing build time and keeping on the same base image will be helpful for speed. We should try to find the right balance between setup time (docker build time, model download and load time) and inference time.
2. The base image for the inference runtime doesn't have gcc or clang, which are required by triton.
3. I've asked [here](https://edstem.org/us/courses/77432/discussion/6658204) what machine we'll be running on. GPU type will be especially important.
