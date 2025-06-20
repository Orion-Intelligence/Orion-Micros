class I2P_COMMANDS:
    S_INIT = 1
    S_FETCH = 2
    S_PROXY = 3


class I2P_PROXY:
    PROXY_URL_HTTP = "http://trusted-micros-i2p:8444"
    PROXY_URL_HTTPS = "http://trusted-micros-i2p:8445"
    SUBSCRIPTION_URLS = ["http://i2p-projekt.i2p/hosts.txt",
                         "http://inr.i2p/hosts.txt",
                         "http://notbob.i2p/hosts.txt",
                         "http://skank.i2p/hosts.txt",
                         "http://stats.i2p/hosts.txt"]
