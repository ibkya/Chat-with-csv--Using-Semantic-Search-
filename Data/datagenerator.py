import random
import pandas as pd
from faker import Faker


fake = Faker()


cities = [
    "Tokyo", "Konya", "Paris", "Gaziantep", "Adana", "İzmir", "Edirne", 
    "Berlin", "Bursa", "Ankara", "Antalya", "New York", "İstanbul", 
    "Diyarbakır", "Londra"
]


def generate_http_method():
    return random.choice(['GET', 'POST', 'PUT', 'DELETE'])


def generate_http_status():
    return random.choice([200, 301, 404, 500, 403])


def generate_url():
    urls = [
        '/home', '/about', '/products', '/contact', '/login', '/signup', 
        '/search', '/cart', '/checkout', '/profile', '/blog', '/services'
    ]
    return random.choice(urls)


def generate_data_size(min_size=100, max_size=5000):
    return random.randint(min_size, max_size)


def generate_log_entry():
    return {
        "IP": fake.ipv4_public(), 
        "Timestamp": fake.date_time_this_year().strftime('%Y-%m-%d %H:%M:%S'),  
        "HTTP_Method": generate_http_method(),
        "URL": generate_url(),
        "HTTP_Status": generate_http_status(),
        "User_Agent": fake.user_agent(),  
        "Referrer": fake.url(),  
        "Request_Size_Bytes": generate_data_size(100, 2000),
        "Response_Size_Bytes": generate_data_size(200, 5000),
        "Session_ID": fake.uuid4(),
        "City": random.choice(cities)  
    }


def generate_fake_log_data(num_entries):
    log_data = [generate_log_entry() for _ in range(num_entries)]
    return pd.DataFrame(log_data)


if __name__ == "__main__":
    num_entries = 1000 
    
    fake_log_data = generate_fake_log_data(num_entries)
    fake_log_data.to_csv('fake_web_traffic_logs_with_cities.csv', index=False)
    print("Sahte veri seti 'fake_web_traffic_logs_with_cities.csv' olarak kaydedildi.")
