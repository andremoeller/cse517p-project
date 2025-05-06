# Overview

Instructions for development.

## Development

Either open project root directory in VSCode and load devcontainer, or build via `make`:

- `make build-runtime` to build the minimal runtime image `evabyte:runtime`
- `make build-dev` to build the development image
- `run-runtime` to run an interactive shell in the minimal runtime image
- `run-dev` to run an interactive shell in the development image
- `shell-dev` to open an additional shell in the development image
- `submit` to test the assets in the submit/ directory using the commands the project grader will run.

The way I'm developing is this: I've started a g2-standard-4, which is what the project grader will use. I allow SSH access onto that host.

I open VS code and use the Remote Development feature to add a remote host over ssh to the google cloud VM. I clone the repository on the remote host, and open the folder as a Dev Container. I develop entirely within the dev container.

To prepare a submission, I run `bash submit.sh`, then scp the submit.zip over to my local machine. To make sure the submission works, you can run `make submit`.

## Notes

Various notes and thoughts that occurred that we may want to keep in mind.

1. According to Ed in [this discussion](https://edstem.org/us/courses/77432/discussion/6630668), timing starts when "docker build -t" _starts_, rather than when it ends. This means optimizing build time and keeping on the same base image will be helpful for speed. We should try to find the right balance between setup time (docker build time, model download and load time) and inference time.
2. The base image for the inference runtime doesn't have gcc or clang, which are required by triton.
3. I've asked [here](https://edstem.org/us/courses/77432/discussion/6658204) what machine we'll be running on. GPU type will be especially important: we'll be on an L4 (24GB).
4. We may be at risk of running OOM on the full dataset, or running really slowly without batching. I made a cheap attempt at batching but ran OOM without trying much further. The test dataset will be 20000 examples according to [this comment](https://edstem.org/us/courses/77432/discussion/6653048).
5. We're at high risk of running out of time for the full test dataset if we don't do at least some amount of batching. We have one hour. Without batching, we do about 2 samples/sec on 20,000 examples, which will take about 3 hours. We may want to create a synthetic test dataset and explore options for reducing memory footprint to iterate on performance. We can quantize in the short run to prepare a valid submission for checkpoint 2.
