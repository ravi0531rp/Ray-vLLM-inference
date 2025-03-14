# Ray-vLLM-inference

## Objective
* Run inference on Multiple GPUs on distributed nodes present on the same WIFI network.
  
## Constraints
* I've got two laptops on the same wifi, and both have Ubuntu 22.
* The first laptop has an RTX 2050, and the second has a decade-old MX150. We will call the former ravi-MSI and the latter ravi-inspiron-7572. (Those are actually the hostnames)

## Steps
* Create python environments with the same version on both the systems.
```sh
conda create -n distributed_infer python=3.10.14
```
* Now install vLLM and Ray on both systems, ensuring they are the same versions.
```sh
pip install vllm
```
* Set up Avahi-daemon on both systems since Ubuntu doesn’t enable mDNS by default
```sh
sudo apt update
sudo apt install avahi-daemon -y

sudo systemctl start avahi-daemon
sudo systemctl enable avahi-daemon

systemctl status avahi-daemon

hostname

```
* Test if both the laptops can talk to each other via hostnames. (This could fail; a simple few searches online would fix that)
```sh
ping ravi-inspiron-7572.local  # try from ravi-msi
ping ravi-msi # try from ravi-inspiron-7572.local

```

* Check if nvidia GPUs are detected on both
```sh
nvidia-smi
```

* Start a ray cluster on ravi-MSI (that has more compute available, hence I chose this)
```sh
ray start --head --port=6380   # default is 6379, but redis was using it in my case for a different task
ray status

```

* Go to second laptop (ravi-inspiron-7572) and connect it to the cluster
```sh
ray start --address=ravi-MSI.local:6380
ray status

```
* In the main laptop, log in to hugginface since our model would be pulled from there.
```sh
pip install --upgrade huggingface_hub
huggingface-cli login

```

* Run the command to start an inference server, which would use both the available GPUs. Make sure to download the models at both the nodes as mentioned [here](https://docs.vllm.ai/en/v0.5.1/serving/distributed_serving.html)
Note -> I have kept dtype float16 since MX150 wont support bfloat16 

```sh
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --tensor-parallel-size 2   --swap-space 2 --dtype float16
```

* Test out inference by
```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "temperature": 0.7
  }'


```







