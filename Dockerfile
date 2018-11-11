FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
RUN apt-get -y update && apt-get -y install cmake
RUN pip install gym pillow atari-py visdom pyopengl matplotlib

RUN git clone https://github.com/neka-nat/player_zoo.git --recursive
ENV ROBOSCHOOL_PATH /workspace/player_zoo/roboschool
WORKDIR /workspace/player_zoo/roboschool
RUN git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision
RUN mkdir -p bullet3/build
RUN cd bullet3/build && \
    cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF .. && \
    make -j4 && make install
WORKDIR /workspace/player_zoo

CMD ['/bin/bash']