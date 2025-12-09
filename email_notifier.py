import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import json
import os

class EmailNotifier:
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file):
        """Load email configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            default_config = {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your_email@gmail.com",
                "sender_password": "your_app_password",
                "owner_email": "car_owner@gmail.com"
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    def send_email(self, subject, body, attachment_path: str = None):
        """Send an email notification. Optionally attach image file."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['owner_email']
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                filename = os.path.basename(attachment_path)
                part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(part)
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['sender_email'], self.config['sender_password'])
            text = msg.as_string()
            server.sendmail(self.config['sender_email'], self.config['owner_email'], text)
            server.quit()
            print(f"Email sent successfully!")
            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
    def send_new_driver_notification(self, driver_name, age, license_no, image_path: str = None):
        """Send notification when a new driver is registered (with optional face image attachment)."""
        subject = "New Driver Registered in Your Car"
        body = f"""
        ALERT: A NEW DRIVER HAS BEEN REGISTERED IN YOUR CAR
        
        Driver Information:
        - Name: {driver_name}
        - Age: {age}
        - License Number: {license_no}
        - Registration Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This person has been added to the system and can now drive your car.
        
        If this is unauthorized, please contact support immediately.
        """
        return self.send_email(subject, body, attachment_path=image_path)
    def send_driver_started_notification(self, driver_name):
        subject = "Driver Started Driving Your Car"
        body = f"""
        NOTIFICATION: A DRIVER HAS STARTED DRIVING YOUR CAR
        
        Driver Information:
        - Name: {driver_name}
        - Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        The driver verification was successful and they are now driving.
        """
        return self.send_email(subject, body)
    def send_drowsiness_alert(self, driver_name, closed_time, image_path: str = None):
        """Send alert when driver is drowsy, with optional face image attachment."""
        subject = "URGENT: Driver Drowsiness Alert"
        body = f"""
        URGENT ALERT: DRIVER IS FEELING SLEEPY
        
        Driver Information:
        - Name: {driver_name}
        - Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Eyes Closed For: {closed_time:.2f} seconds
        
        The driver's eyes have been closed for more than 3 seconds.
        The system has been alerted with a loud noise.
        
        Please ensure the driver stops immediately and takes a break.
        """
        return self.send_email(subject, body, attachment_path=image_path)
