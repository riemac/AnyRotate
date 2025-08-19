---
trigger: model_decision
description: IsaacLab开发规则
---

## IsaacLab开发调试规则

当在开发和调试过程中遇到以下情况时，你应按照以下规则处理：

### 1. 工作区结构说明

当前项目采用多根工作区结构，包含以下两个主要目录：

1. **leaphand项目目录**：当前开发的项目根目录，包含所有自定义代码和配置
2. **IsaacLab项目目录**：核心框架目录，包含IsaacLab的所有基础功能和API

在开发和调试过程中，应充分利用IsaacLab提供的功能，同时保持项目代码的独立性，避免直接修改IsaacLab核心代码。

### 2. IsaacLab API或概念等不清楚时的处理流程

1. **优先查阅IsaacLab源码和文档**
   - 在IsaacLab项目目录中查找相关文档和示例代码
   - 可在以下路径查找相关文档：
     - `docs/`: 官方文档
     - `source/`: 源代码目录，包含详细的代码注释
     - `examples/`: 示例代码
     - `scripts/tutorials`: 官方教程示例

2. **使用context7工具查询**
   - 使用`mcp_context7_resolve-library-id`和`mcp_context7_get-library-docs`工具查询官方文档
   - 如usda文件的层级结构或术语不了解，则可在context7中搜索usd相关信息；IsaacLab环境配置API不明确，可在context7中搜索isaaclab相关信息。其他同理
   - 通过这种方式可以获取最准确、最新的API信息，避免产生幻觉或使用不存在的API

3. **其次使用网络搜索**
   - 如果context7工具未能解决疑问，可以使用网络搜索工具进一步查找相关信息
   - 搜索时应优先考虑官方文档、GitHub仓库和权威技术博客

4. **避免产生幻觉**
   - 不要凭空编造API或功能
   - 不要假设存在未验证的函数或类
   - 始终以查询到的官方文档为准

## 开发实践建议

- 在编写和调试修改代码前，先查询相关模块的文档和示例代码
- 参考IsaacLab官方提供的环境配置和实现方式
- 保持代码风格与IsaacLab项目一致
- 提供丰富且阅读友好的注释
- 充分利用IsaacLab提供的工具和框架功能