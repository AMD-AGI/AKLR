# The image of a Primus-SaFE GithubRunner pod that runs ci.yaml's GPU jobs: one of stack.json's
# images plus the GitHub Actions runner, and nothing else. What a job installs and runs is in ci.yaml.
#   docker build --build-arg BASE=<stack.json sglang_image or vllm_image> -f .github/runner/Dockerfile .github/runner
ARG BASE
FROM ${BASE}
ARG BASE
ARG RUNNER_VERSION=2.338.2
# ci.yaml compares this with stack.json: a pod keeps the image its workload was created with.
ENV AKLR_RUNNER_BASE=${BASE}
USER root
# Primus-SaFE starts /home/runner/config.sh and run.sh as uid 1001.
RUN useradd --uid 1001 --create-home --home-dir /home/runner runner \
 && curl -fsSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    | tar xz -C /home/runner \
 && /home/runner/bin/installdependencies.sh \
 && apt-get install -y --no-install-recommends jq \
 && rm -rf /var/lib/apt/lists/* \
 && chown -R 1001 /home/runner
USER 1001
WORKDIR /home/runner
# The stack images start a server; the platform supplies the runner's command.
ENTRYPOINT []
