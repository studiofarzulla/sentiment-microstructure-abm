"""
Base Agent Architecture
Defines the interface for all autonomous agents in the simulation.
"""

from abc import ABC, abstractmethod
import logging
import asyncio
import signal
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract Base Class for autonomous agents.
    
    Handles:
    - Lifecycle management (start, stop)
    - Event loop integration
    - Signal handling
    - Basic state management
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(f"agent.{name}")
        
    async def start(self):
        """Start the agent's main loop."""
        self.is_running = True
        self.logger.info(f"Starting agent: {self.name}")
        
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                # Windows support or non-main thread
                pass
                
        try:
            await self.run()
        except asyncio.CancelledError:
            self.logger.info("Agent task cancelled")
        except Exception as e:
            self.logger.error(f"Agent crashed: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def stop(self):
        """Signal the agent to stop."""
        self.logger.info(f"Stopping agent: {self.name}")
        self.is_running = False
        self._shutdown_event.set()

    @abstractmethod
    async def run(self):
        """Main execution logic. Must be implemented by subclasses."""
        pass

    async def cleanup(self):
        """Cleanup resources. Can be overridden."""
        self.logger.info("Cleaning up resources...")

    async def wait_until_shutdown(self):
        """Wait until the shutdown signal is received."""
        await self._shutdown_event.wait()
