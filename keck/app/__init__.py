"""
CUDA 代码验证服务

模块结构:
- config.py: 配置管理
- schemas.py: 数据模型
- ray_init.py: Ray 初始化
- main.py: FastAPI 路由入口
- services/: 业务逻辑
  - compiler.py: CUDA 编译（CPU）
  - executor.py: 验证执行（GPU）
  - pipeline.py: 流水线编排
- utils/: 工具函数
  - ast_utils.py: AST 重写
  - message_parser.py: 消息解析
- templates/: 验证模板脚本
  - test_cuda.py: CUDA vs PyTorch
  - test_torch.py: PyTorch vs PyTorch
"""
