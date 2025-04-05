# Use official Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    postgresql \
    postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Specify volume mount point
# VOLUME /Users/nandhakishorecs/Documents/IITM/Jan_2025/DA5402/DA5402_AI_Application
# VOLUME DA5402_AI_Application

# Expose ports (PostgreSQL and custom port)
EXPOSE 5432 18000

# Create PostgreSQL data directory and set permissions
RUN mkdir -p /var/lib/postgresql/data && \
    chown postgres:postgres /var/lib/postgresql/data

# Copy initialization script
COPY init_sql.sh /init_sql.sh
RUN chmod +x /init_sql.sh

# Start PostgreSQL and keep container running
CMD ["/init_sql.sh"]