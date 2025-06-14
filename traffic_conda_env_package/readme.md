# Conda 离线环境迁移指南

本指南用于将 `traffic` Conda 虚拟环境迁移至一台离线计算机，并确保能够使用 `conda activate traffic` 命令正常激活。

## 操作步骤

### 1. 准备工作

- 确保离线计算机上已经安装了 Conda (Miniconda 或 Anaconda)。
- 将本文件夹 (`traffic_conda_env_package`) 完整地拷贝到离线计算机的任意位置（例如用户主目录 `~`）。

### 2. 确定 Conda 安装路径

在离线计算机上打开终端，执行以下命令找到 Conda 的安装路径：

```bash
conda info --base
```

该命令会输出 Conda 的根目录路径，例如 `/opt/miniconda3` 或 `/home/user/anaconda3`。请记下这个路径，下文将以 `/path/to/conda` 作为示例。

### 3. 解压环境到 Conda 目录

将 `traffic.tar.gz` 文件解压到 Conda 的环境目录 (`envs`) 下。

```bash
# 首先，在 Conda 的 envs 目录下创建一个名为 traffic 的空文件夹
mkdir -p /path/to/conda/envs/traffic

# 然后，解压 tar.gz 文件到该目录
# (请确保 traffic.tar.gz 在当前路径下，或使用其绝对路径)
tar -xzf traffic.tar.gz -C /path/to/conda/envs/traffic
```
**注意**：请将上面的 `/path/to/conda` 替换为您在第二步中找到的真实路径。

### 4. 修复环境路径 (关键步骤)

解压完成后，需要进入该环境并执行 `conda-unpack` 脚本来修复包内的路径依赖。

```bash
# 1. 使用 source 命令临时激活环境
source /path/to/conda/envs/traffic/bin/activate

# 2. 执行修复脚本
conda-unpack

# 3. 退出临时激活的环境
conda deactivate
```
**注意**：同样，请将 `/path/to/conda` 替换为真实路径。

### 5. 验证

完成以上所有步骤后，您就可以在离线计算机上通过标准方式激活和使用 `traffic` 环境了。

```bash
conda activate traffic
```

如果命令执行后，终端提示符前出现 `(traffic)` 字样，则代表环境已成功迁移并激活！
