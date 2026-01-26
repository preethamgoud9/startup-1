# 🐳 How to Run with Docker

This system is now dockerized, making it easy to run on any machine with Docker installed.

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

## Setup and Run

1.  **Build and Start the System**:
    Open your terminal in the root directory and run:
    ```bash
    docker compose up --build -d
    ```

2.  **Access the Application**:
    - **Frontend**: [http://localhost](http://localhost) (Port 80)
    - **Backend API**: [http://localhost:8000](http://localhost:8000)
    - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

3.  **Stop the System**:
    ```bash
    docker compose down
    ```

## ⚠️ Important Notes

### Webcam Access
Docker containers on macOS have limited direct access to host hardware like webcams. 
- **On Linux**: Direct access is possible by uncommenting the `devices` section in `docker-compose.yml`.
- **On macOS/Windows**: It is recommended to use an **RTSP stream** (e.g., from an IP camera or a phone app like "IP Webcam") and configure the URL in `backend/config.yaml`.

### Persistent Data
All attendance records and face embeddings are saved in the `backend/data` folder on your host machine, so they won't be lost when you stop the containers.
