# NoCodeML Platform 🚀

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)

---

## 📸 Preview

![NoCodeML Platform Landing Page](./screenshots/landing-page.png)
*NoCodeML Platform - Your gateway to no-code machine learning*

---

## 🎯 What We Built

NoCodeML is an end-to-end machine learning platform that removes the coding barrier from ML development. Users can:
- Upload datasets (CSV, Excel, Parquet)
- Perform automated exploratory data analysis with interactive visualizations
- Select and configure ML models from 8 powerful algorithms
- Train models asynchronously with real-time progress tracking
- Make predictions on new data
- Compare model performance across different algorithms

**Tech Stack:** FastAPI + React + PostgreSQL + Redis + Celery + Docker  
**Models:** 4 Classification + 4 Regression (Logistic/Linear, Random Forest, XGBoost, LightGBM)

### Why This Matters
Machine learning shouldn't require a CS degree. NoCodeML makes ML accessible to:
- Business analysts exploring data patterns
- Students learning ML concepts
- Researchers testing hypotheses quickly
- Anyone with data and questions to answer

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (with npm or bun)

### Setup (5 minutes)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd V2_NoCodeML

# 2. Backend - Start all services (API, PostgreSQL, Redis, Celery Worker)
cd Backend
cp .env.example .env
docker-compose up -d
docker-compose exec fastapi_app alembic upgrade head

# 3. Frontend - Install and run
cd ../Frontend
npm install  # or: bun install
cp .env.example .env
npm run dev  # or: bun run dev
```

### Access Points
- **Application:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs
- **Backend:** http://localhost:8000

---

## ✨ Core Features

### 1️⃣ Dataset Management
- **Multi-format Support:** Upload CSV, Excel (.xlsx/.xls), and Parquet files
- **Smart Metadata Extraction:** Automatic detection of data types, row/column counts
- **Quick Preview:** View dataset samples directly in the browser
- **Full CRUD Operations:** Rename, update, delete, and manage multiple datasets

### 2️⃣ Exploratory Data Analysis (EDA)
- **Automated Statistics:** Mean, median, std deviation, quartiles for all numeric columns
- **Missing Value Analysis:** Identify and visualize data quality issues
- **Correlation Matrix:** Interactive heatmap showing feature relationships
- **Smart Visualizations:**
  - Distribution plots (histograms)
  - Box plots for outlier detection
  - Scatter plots for bivariate analysis
  - Powered by Plotly (backend) and Recharts (frontend)
- **Outlier Detection:** IQR-based statistical outlier identification

### 3️⃣ Experiment Workflow
- **Project Organization:** Create experiments linked to specific datasets
- **Version Control:** Duplicate experiments to test different configurations
- **Configuration Persistence:** Save and load model configurations
- **5-Step Guided Workflow:** Analysis → Config → Training → Results → Prediction

### 4️⃣ Machine Learning Models (8 Algorithms)

**Classification** (4 models)
- **Logistic Regression** - Fast linear classifier for binary/multi-class problems
- **Random Forest** - Ensemble of decision trees with feature importance
- **XGBoost** - Gradient boosting with regularization and high accuracy
- **LightGBM** - Ultra-fast gradient boosting optimized for speed

**Regression** (4 models)
- **Linear Regression** - Simple and interpretable linear model
- **Random Forest** - Robust ensemble regressor with feature importance
- **XGBoost** - High-accuracy gradient boosting for regression
- **LightGBM** - Memory-efficient and fast gradient boosting

### 5️⃣ Model Training System
- **Asynchronous Processing:** Celery-based distributed task queue
- **Real-time Status:** Live progress tracking with job status updates
- **Multi-model Training:** Train multiple algorithms simultaneously
- **Smart Hyperparameters:** Pre-configured optimal defaults for each model
- **Comprehensive Metrics:**
  - **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - **Regression:** R², MAE, MSE, RMSE
- **Model Persistence:** Automatic saving of trained models with joblib
- **Feature Importance:** Supported for Random Forest, XGBoost, and LightGBM models

### 6️⃣ Prediction Engine
- **Single Predictions:** Real-time predictions on individual data points
- **Batch Processing:** Upload CSV files for bulk predictions
- **Result Export:** Download predictions as CSV files
- **Confidence Scores:** Probability estimates for classification tasks

### 7️⃣ User Experience
- **Authentication:** JWT-based secure user system
- **Modern UI:** Built with shadcn/ui components and Tailwind CSS
- **Responsive Design:** Works seamlessly on desktop and mobile
- **Interactive Charts:** Dynamic, zoomable visualizations
- **Real-time Feedback:** Toast notifications and progress indicators

---

## 🏗️ Technical Architecture

### Backend Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI (Python 3.11) | High-performance async REST API |
| Database | PostgreSQL 15 | Persistent storage for users, datasets, experiments |
| ORM | SQLAlchemy (async) | Database abstraction layer |
| Migrations | Alembic | Database schema version control |
| Task Queue | Celery + Redis | Asynchronous ML model training |
| ML Libraries | scikit-learn, XGBoost, LightGBM | 8 optimized ML algorithms |
| Data Processing | pandas, numpy, openpyxl, pyarrow | Dataset handling and transformations |
| Visualization | Plotly | Server-side chart generation |
| Authentication | JWT (python-jose) | Secure user authentication |
| Containerization | Docker + Docker Compose | Easy deployment and scaling |

### Frontend Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 18 + TypeScript | Type-safe component-based UI |
| Build Tool | Vite | Fast development and optimized builds |
| UI Library | shadcn/ui (Radix UI) | Accessible, customizable components |
| Styling | Tailwind CSS | Utility-first responsive design |
| State Management | React Context API | Global state for auth, experiments |
| Data Fetching | TanStack Query | Caching and server state management |
| Routing | React Router v6 | Client-side navigation |
| Charts | Recharts | Interactive data visualizations |

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
│                    (React + TypeScript)                     │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Auth API   │  │ Dataset API  │  │ Training API │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────┬─────────────────┬──────────────────┬──────────────┘
         │                 │                  │
         ▼                 ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
│   PostgreSQL    │ │    Redis     │ │  Celery Worker  │
│   (Database)    │ │ (Task Queue) │ │ (ML Training)   │
└─────────────────┘ └──────────────┘ └─────────────────┘
```

### Project Structure

```
V2_NoCodeML/
├── Backend/                  # FastAPI backend application
│   ├── app/
│   │   ├── api/             # API route handlers
│   │   │   ├── auth.py      # Authentication endpoints
│   │   │   ├── datasets.py  # Dataset management
│   │   │   ├── eda.py       # Exploratory data analysis
│   │   │   ├── experiments.py  # Experiment CRUD
│   │   │   ├── models.py    # ML model catalog
│   │   │   ├── training.py  # Model training
│   │   │   └── predictions.py  # Prediction endpoints
│   │   ├── core/            # Core configuration
│   │   │   ├── config.py    # App settings
│   │   │   ├── security.py  # Auth utilities
│   │   │   └── model_cache.py  # Model caching
│   │   ├── db/              # Database configuration
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── schemas/         # Pydantic validation schemas
│   │   ├── services/        # Business logic layer
│   │   ├── worker/          # Celery tasks
│   │   └── main.py          # Application entry point
│   ├── alembic/             # Database migrations
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile
│   └── docker-compose.yaml
│
└── Frontend/                # React frontend application
    ├── src/
    │   ├── components/      # Reusable UI components
    │   │   ├── datasets/    # Dataset components
    │   │   ├── experiments/ # Experiment components
    │   │   ├── playground/  # ML workflow components
    │   │   └── ui/          # shadcn/ui components
    │   ├── contexts/        # React Context providers
    │   ├── hooks/           # Custom React hooks
    │   ├── pages/           # Page components
    │   ├── services/        # API services
    │   └── types/           # TypeScript type definitions
    ├── package.json
    └── vite.config.ts
```

---

## 🛠️ Development Guide

### Backend Commands

```bash
# Start all services (API, PostgreSQL, Redis, Celery)
cd Backend
docker-compose up -d

# View logs
docker-compose logs -f fastapi_app
docker-compose logs -f celery_worker

# Database migrations with Alembic
docker-compose exec fastapi_app alembic upgrade head              # Apply all migrations
docker-compose exec fastapi_app alembic revision --autogenerate -m "add_column"  # Create new migration
docker-compose exec fastapi_app alembic downgrade -1              # Rollback last migration
docker-compose exec fastapi_app alembic current                   # Show current revision
docker-compose exec fastapi_app alembic history                   # Show migration history

# Access container shell
docker-compose exec fastapi_app bash

# Stop all services
docker-compose down
```

### Frontend Commands

```bash
cd Frontend

# Development server (hot reload)
npm run dev          # or: bun run dev

# Production build
npm run build        # or: bun run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Environment Configuration

**Backend** (`Backend/.env`):
```env
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DB=nocodeml_db
DATABASE_URL=postgresql+psycopg://myuser:mysecretpassword@postgres:5432/nocodeml_db
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
SECRET_KEY=your-secret-key-change-in-production
```

**Frontend** (`Frontend/.env`):
```env
VITE_API_URL=http://localhost:8000
```

---

## 📡 API Reference

Full interactive API documentation: **http://localhost:8000/docs**

### Core Endpoints

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| **Auth** | `/api/v1/auth/register` | POST | Register new user |
| | `/api/v1/auth/login` | POST | Login and get JWT token |
| **Datasets** | `/api/v1/datasets` | GET, POST | List/upload datasets |
| | `/api/v1/datasets/{id}` | GET, PUT, DELETE | Manage dataset |
| | `/api/v1/datasets/{id}/preview` | GET | Preview dataset rows |
| **Experiments** | `/api/v1/experiments` | GET, POST | List/create experiments |
| | `/api/v1/experiments/{id}` | GET, PUT, DELETE | Manage experiment |
| | `/api/v1/experiments/{id}/duplicate` | POST | Duplicate experiment |
| **EDA** | `/api/v1/eda/{dataset_id}/summary` | GET | Get statistics & correlations |
| | `/api/v1/eda/plot` | POST | Generate visualization |
| **Models** | `/api/v1/models` | GET | List all 8 ML models |
| | `/api/v1/models/{task_type}` | GET | Filter by classification/regression |
| **Training** | `/api/v1/training/start` | POST | Start async training jobs |
| | `/api/v1/training/runs/{job_id}` | GET | Get training status & results |
| **Predictions** | `/api/v1/predictions/batch` | POST | Batch predictions from CSV |
| | `/api/v1/predictions/{batch_id}` | GET | Get prediction results |

---

## 🗄️ Database Schema & Migrations

### Alembic Migrations

The project uses **Alembic** for database schema version control, ensuring smooth schema evolution across environments.

#### Current Migrations

1. **`001_create_datasets_table.py`** - User datasets table
   - Stores uploaded dataset metadata (name, file path, rows, columns)
   - User-scoped with foreign key to users table

2. **`002_create_experiments_table.py`** - ML experiments table
   - Links experiments to datasets
   - Stores experiment configuration (features, target, task type)

3. **`003_create_training_tables.py`** - Training infrastructure
   - Training jobs table (status, model type, hyperparameters)
   - Training results table (metrics, feature importance)

4. **`004_create_training_runs_table.py`** - Enhanced training tracking
   - Detailed job execution tracking
   - Training logs and progress monitoring

5. **`005_create_prediction_batches_table.py`** - Prediction system
   - Batch prediction management
   - Prediction results storage

#### Migration Commands

```bash
# Initialize database (first time setup)
docker-compose exec fastapi_app alembic upgrade head

# Create new migration after model changes
docker-compose exec fastapi_app alembic revision --autogenerate -m "add_new_column"

# Apply migrations
docker-compose exec fastapi_app alembic upgrade head

# Rollback last migration
docker-compose exec fastapi_app alembic downgrade -1

# View current database version
docker-compose exec fastapi_app alembic current

# View migration history
docker-compose exec fastapi_app alembic history --verbose
```

### Database Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `users` | User authentication | email, hashed_password, created_at |
| `datasets` | Uploaded datasets | name, file_path, num_rows, num_columns, user_id |
| `experiments` | ML experiments | name, dataset_id, config (JSONB), user_id |
| `training_jobs` | Training tasks | status, model_type, experiment_id |
| `training_results` | Model metrics | accuracy, precision, f1_score, job_id |
| `training_runs` | Run tracking | start_time, end_time, error_message |
| `training_logs` | Progress logs | message, metrics_json, timestamp |
| `prediction_batches` | Predictions | input_file_path, output_file_path, status |

---

## 🎓 Technical Highlights

### What We Built
- **Full ML Pipeline:** Complete workflow from data upload to predictions
- **8 Production-Ready Models:** 4 classification + 4 regression algorithms (Logistic/Linear Regression, Random Forest, XGBoost, LightGBM)
- **Async Training:** Celery + Redis for non-blocking model training
- **Type-Safe Frontend:** Full TypeScript coverage with proper interfaces
- **Interactive Viz:** Server-side Plotly, client-side Recharts for data exploration
- **Microservices Architecture:** Separate containers for API, worker, database, cache

### Challenges Overcome
1. **Long-running Tasks:** Implemented Celery task queue to handle training jobs that can take minutes
2. **State Synchronization:** Real-time status updates between training worker and UI via polling
3. **Type Safety:** Consistent TypeScript types across 50+ React components
4. **Data Validation:** Robust Pydantic schemas preventing bad data from reaching ML pipeline
5. **Model Persistence:** Efficient joblib serialization and retrieval of trained models

### Key Architecture Decisions
| Decision | Rationale |
|----------|-----------|
| FastAPI over Flask | Native async support, automatic OpenAPI docs, better performance |
| Celery for training | Prevents API timeouts on long-running ML tasks |
| React Context API | Simpler than Redux for our scope; sufficient for auth/experiment state |
| Docker Compose | Single-command dev environment with all services |
| PostgreSQL | ACID compliance for experiment/training data |
| shadcn/ui | Accessible components, full customization control |

---
