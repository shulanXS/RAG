# =============================================================================
# Enterprise RAG System — Makefile
# =============================================================================
# 用法: make <target>
#
# 前置条件:
#   - Python >= 3.11
#   - Docker & Docker Compose (用于容器化部署)
#   - Node.js >= 18 (用于前端开发)
# =============================================================================

.PHONY: help install dev test lint clean \
	up down restart logs status \
	backend-uvicorn frontend-dev \
	ingest eval \
	build build-backend build-frontend

# =============================================================================
# 变量
# =============================================================================
PYTHON := python3
PYTEST := pytest
FRONTEND_PORT := 3000
BACKEND_PORT := 8000

# =============================================================================
# 帮助信息
# =============================================================================
help:
	@echo "Enterprise RAG System — 可用命令"
	@echo ""
	@echo "  安装与运行"
	@echo "    install         安装 Python 依赖"
	@echo "    dev             启动后端开发服务器 (uvicorn)"
	@echo "    frontend-dev    启动前端开发服务器"
	@echo "    test            运行单元测试"
	@echo "    lint            代码风格检查"
	@echo ""
	@echo "  Docker 容器"
	@echo "    up              启动所有服务 (docker-compose)"
	@echo "    down            停止所有服务"
	@echo "    restart         重启所有服务"
	@echo "    logs            查看服务日志"
	@echo "    status          查看服务状态"
	@echo ""
	@echo "  数据与评估"
	@echo "    ingest          运行文档索引脚本"
	@echo "    eval            运行系统评估"
	@echo ""
	@echo "  构建"
	@echo "    build           构建所有 Docker 镜像"
	@echo "    build-backend   仅构建后端镜像"
	@echo "    build-frontend  仅构建前端镜像"
	@echo ""
	@echo "  清理"
	@echo "    clean           清理缓存和临时文件"

# =============================================================================
# 安装
# =============================================================================
install:
	@echo "安装 Python 依赖..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo "安装完成"

# =============================================================================
# 开发
# =============================================================================
dev:
	@echo "启动后端服务 (http://localhost:$(BACKEND_PORT))..."
	uvicorn backend.main:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload

frontend-dev:
	@echo "启动前端服务 (http://localhost:$(FRONTEND_PORT))..."
	cd frontend && npm install && npm run dev

# =============================================================================
# 测试
# =============================================================================
test:
	@echo "运行测试..."
	$(PYTEST) tests/ -v --tb=short

test-cov:
	@echo "运行测试并生成覆盖率报告..."
	$(PYTEST) tests/ -v --cov=backend --cov-report=html --cov-report=term

# =============================================================================
# 代码检查
# =============================================================================
lint:
	@echo "运行代码检查..."
	cd backend && $(PYTHON) -m ruff check . || true
	$(PYTHON) -m ruff check . || true

format:
	@echo "格式化代码..."
	$(PYTHON) -m ruff format .

# =============================================================================
# Docker 容器管理
# =============================================================================
up:
	@echo "启动所有服务..."
	docker-compose up -d
	@echo "服务已启动: http://localhost:$(FRONTEND_PORT)"

down:
	@echo "停止所有服务..."
	docker-compose down

restart:
	@echo "重启所有服务..."
	docker-compose restart

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

status:
	docker-compose ps

# =============================================================================
# 构建 Docker 镜像
# =============================================================================
build:
	docker-compose build

build-backend:
	docker-compose build backend

build-frontend:
	docker-compose build frontend

# =============================================================================
# 数据与评估
# =============================================================================
ingest:
	@echo "运行文档索引..."
	$(PYTHON) scripts/ingest.py --source data/sample_docs

eval:
	@echo "运行系统评估..."
	$(PYTHON) scripts/eval.py

# =============================================================================
# 清理
# =============================================================================
clean:
	@echo "清理缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "清理完成"
