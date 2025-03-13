import struct
import datetime

class ObjectId:
    """Minimal ObjectId implementation for BSON compatibility."""

    def __init__(self, oid: str):
        """Initialize with a hex string ObjectId (24 characters)."""
        if len(oid) != 24:
            raise ValueError("ObjectId must be a 24-character hex string")
        self.oid = oid

    @property
    def generation_time(self):
        """Extract timestamp from ObjectId and return a UTC datetime."""
        timestamp = int(self.oid[:8], 16)  # First 4 bytes = timestamp
        return datetime.datetime.utcfromtimestamp(timestamp)

# Example usage:
if __name__ == "__main__":
    oid = ObjectId("60d5f9b2d4f14b3a9c8e2e32")
    print(oid.generation_time)  # Should return the creation time
