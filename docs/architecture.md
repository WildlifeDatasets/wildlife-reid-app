# System Architecture

The Wildlife ReID App is built as a microservices application orchestrated using Docker Compose. It separates the user-facing web interface from the resource-intensive machine learning tasks required for image analysis.

## High-Level Overview

The system consists of three main logic components:
1.  **API (Django):** Handles user interactions, data management, and orchestration.
2.  **Taxon Worker:** Performs species classification (ML inference).
3.  **Identification Worker:** Performs individual re-identification (ML inference).

These components communicate asynchronously via a message broker (RabbitMQ) and share data through a database and file storage.

## Service Components

### 1. Web API (`api`)
*   **Technology:** Python, Django, Django REST Framework.
*   **Role:**
    *   Serves the web frontend and REST API.
    *   Manages user authentication (including OAuth).
    *   Handles file uploads.
    *   Dispatches tasks to workers via Celery.
*   **Dependencies:** Connects to the Main Database, Redis, and RabbitMQ.

### 2. Workers
These services run in the background, typically on hardware with GPU acceleration.

*   **Taxon Worker (`taxon_worker`):**
    *   Receives images from the queue.
    *   Classifies the species/taxon of the animal in the image.
    *   Returns the classification result to the API.
*   **Identification Worker (`identification_worker`):**
    *   Receives cropped images of specific individuals.
    *   Computes embeddings (feature vectors) to identify unique individuals.
    *   Uses its own dedicated database (`identification_worker_db`) to store vector embeddings for fast retrieval.

### 3. Infrastructure

*   **Nginx (`nginx`):**
    *   Acts as a reverse proxy.
    *   Serves static files (images, CSS, JS).
    *   Forwards application requests to the API container.
*   **Message Broker (`broker`):**
    *   **Technology:** RabbitMQ.
    *   Distributes tasks from the API to the workers.
*   **Cache & Result Backend (`redis`):**
    *   **Technology:** Redis.
    *   Stores Celery task results and handles application caching.
*   **Databases:**
    *   **Main DB (`db`):** PostgreSQL. Stores application data (users, metadata, file paths).
    *   **Identification DB (`identification_worker_db`):** PostgreSQL. Dedicated to the identification worker for efficient handling of identification data.

## Data Flow

1.  **Upload:** A user uploads a batch of images via the Web UI.
2.  **Storage:** Images are stored in the shared volume (`api-data` or external storage).
3.  **Task Dispatch:** The API creates metadata in the Main DB and sends a classification task to the `broker`.
4.  **Classification:** The `taxon_worker` picks up the task, processes the image (GPU), and updates the classification result in the Main DB.
5.  **Identification:** If confirmed, the API sends a task to the `identification_worker`.
6.  **Matching:** The `identification_worker` compares the image against known individuals in `identification_worker_db` and returns potential matches.

## Storage Architecture

The application uses several Docker volumes to persist data:

*   `api-data`: Stores uploaded media files.
*   `db-data`: Persists the main PostgreSQL database.
*   `identification-worker-db-data`: Persists the identification worker's database.
*   `caid_import`: A bind-mount used for importing large datasets from the local filesystem.
