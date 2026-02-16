#!/usr/bin/env bash
# Start both backend and frontend servers
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
NC='\033[0m'

# Check if setup has been run
if [ ! -d "$ROOT_DIR/backend/venv" ]; then
    echo "First run detected. Running setup..."
    bash "$ROOT_DIR/setup.sh"
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo "Frontend dependencies missing. Running setup..."
    bash "$ROOT_DIR/setup.sh"
fi

cleanup() {
    echo ""
    echo -e "${GREEN}Shutting down...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT INT TERM

# Start backend
echo -e "${GREEN}Starting backend server...${NC}"
cd "$ROOT_DIR/backend"
source venv/bin/activate
python main.py &
BACKEND_PID=$!

# Start frontend
echo -e "${GREEN}Starting frontend server...${NC}"
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  App is running!${NC}"
echo -e "${GREEN}  Frontend: http://localhost:5173${NC}"
echo -e "${GREEN}  Backend:  http://localhost:8000${NC}"
echo -e "${GREEN}  API Docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

wait
