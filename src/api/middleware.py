import time
import logging
from fastapi import Request

logger = logging.getLogger("api_logger")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"{request.client.host} | "
        f"{request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"{process_time:.2f} ms"
    )

    return response