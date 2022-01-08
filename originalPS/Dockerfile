FROM ubuntu
RUN apt-get update  \
    && apt-get install -y python3 \
                        python3-dev \
                        python3-pip \
                        net-tools \
                        iputils-ping \
                        openssh-server\
                        vim\
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*
    

COPY . /daml 

WORKDIR /daml

RUN pip3 install -r requirements.txt

CMD ["bin/bash"]
