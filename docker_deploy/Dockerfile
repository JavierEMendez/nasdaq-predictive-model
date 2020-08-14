FROM lambci/lambda:build-python3.7
WORKDIR /build
RUN rm -rf *
COPY deploy_requirements.txt .
RUN mkdir pkgs && pip3 install -r deploy_requirements.txt -t pkgs/
CMD ["/bin/bash"]