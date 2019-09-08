def log_response(logger, response):

    logger.debug(response.get_data().decode().strip())
