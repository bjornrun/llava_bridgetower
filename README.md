# llava_bridgetower
Code for a combined Llava and BridgeTower REST API server for Intel Gaudi 2 

[Read about how to use it here](https://runaker.medium.com/enhancing-mixtral-8x22b-with-vision-an-on-premises-solution-for-intel-gaudi-21222c286194)

NOTE: tested on SynapseAI 1.16.2

# Setup:
```
git clone https://github.com/huggingface/optimum-habana/
cd optimum-habana/
```
# Start the container with the correct environment
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v $(pwd):/workspace -v /data/$USER:/root vault.habana.ai/gaudi-docker/1.16.2/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
cd /workspace/
pip install -e .
pip install flask
```
# Start server
```
python llavabridge.py --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf --use_hpu_graphs --bf16
```
