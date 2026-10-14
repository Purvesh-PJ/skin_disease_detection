import logging
from gunicorn.glogging import Logger

class QuietHealthcheckLogger(Logger):
    """
    Suppresses repetitive internal Render/Go-http-client healthcheck pings (HEAD /)
    from access logs, while keeping all user traffic, API endpoints (/predict, /history),
    and error logs crystal clear.
    """
    def access(self, resp, req, environ, request_time):
        # Ignore healthcheck pings to root (HEAD / or GET /)
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        user_agent = environ.get('HTTP_USER_AGENT', '')

        if method in ('HEAD', 'GET') and path == '/' and 'Go-http-client' in user_agent:
            return

        super().access(resp, req, environ, request_time)
