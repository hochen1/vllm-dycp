export container_name="community"
NODE_NAME=""



nerdctl run -td \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --privileged=true \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci8 \
        --device=/dev/davinci9 \
        --device=/dev/davinci10 \
        --device=/dev/davinci11 \
        --device=/dev/davinci12 \
        --device=/dev/davinci13 \
        --device=/dev/davinci14 \
        --device=/dev/davinci15 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -e NODE_NAME=${NODE_NAME} \
        -v <your-workspace>:<your-workspace> \
        -v /usr/libexec:/usr/libexec \
        -v /root:/root \
        -v /etc:/etc \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        --cpus 300 \
        --memory 2000g \
        --shm-size 2000g \
        --net=host \
        --name ${container_name} \
        quay.io/ascend/vllm-ascend:v0.13.0rc1-openeuler \
	bash

