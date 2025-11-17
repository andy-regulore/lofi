"""
Email Marketing Integration

Integrates with Mailchimp and other email services for:
- Newsletter automation
- New release announcements
- Lead magnet delivery (free sample packs)
- Subscriber engagement

Author: Claude
License: MIT
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path


class MailchimpManager:
    """Mailchimp email marketing automation."""

    def __init__(self, api_key: str, server_prefix: str, list_id: str):
        """
        Initialize Mailchimp manager.

        Args:
            api_key: Mailchimp API key
            server_prefix: Server prefix (e.g., 'us1', 'us2')
            list_id: Mailchimp audience/list ID
        """
        self.api_key = api_key
        self.server_prefix = server_prefix
        self.list_id = list_id
        self.base_url = f"https://{server_prefix}.api.mailchimp.com/3.0"

    def add_subscriber(
        self,
        email: str,
        first_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict:
        """
        Add new subscriber to list.

        Args:
            email: Email address
            first_name: Subscriber first name
            tags: List of tags

        Returns:
            Subscription result
        """
        url = f"{self.base_url}/lists/{self.list_id}/members"

        data = {
            'email_address': email,
            'status': 'subscribed'
        }

        if first_name:
            data['merge_fields'] = {'FNAME': first_name}

        if tags:
            data['tags'] = tags

        headers = {
            'Authorization': f'apikey {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, json=data, headers=headers)

            if response.status_code in [200, 201]:
                return {
                    'status': 'success',
                    'email': email,
                    'message': 'Subscriber added'
                }
            else:
                return {
                    'status': 'error',
                    'message': response.json().get('detail', 'Unknown error')
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def create_campaign(
        self,
        subject: str,
        preview_text: str,
        content: str,
        from_name: str = 'LoFi AI',
        reply_to: str = 'hello@lofibeats.ai'
    ) -> Dict:
        """
        Create email campaign.

        Args:
            subject: Email subject
            preview_text: Preview text
            content: HTML content
            from_name: Sender name
            reply_to: Reply-to email

        Returns:
            Campaign info
        """
        url = f"{self.base_url}/campaigns"

        data = {
            'type': 'regular',
            'recipients': {
                'list_id': self.list_id
            },
            'settings': {
                'subject_line': subject,
                'preview_text': preview_text,
                'from_name': from_name,
                'reply_to': reply_to,
                'title': f"Campaign: {subject}"
            }
        }

        headers = {
            'Authorization': f'apikey {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            # Create campaign
            response = requests.post(url, json=data, headers=headers)

            if response.status_code in [200, 201]:
                campaign = response.json()
                campaign_id = campaign['id']

                # Set content
                content_url = f"{self.base_url}/campaigns/{campaign_id}/content"
                content_data = {'html': content}

                requests.put(content_url, json=content_data, headers=headers)

                return {
                    'status': 'success',
                    'campaign_id': campaign_id,
                    'message': 'Campaign created'
                }
            else:
                return {
                    'status': 'error',
                    'message': response.json().get('detail', 'Unknown error')
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def send_campaign(self, campaign_id: str) -> Dict:
        """Send campaign immediately."""
        url = f"{self.base_url}/campaigns/{campaign_id}/actions/send"

        headers = {
            'Authorization': f'apikey {self.api_key}'
        }

        try:
            response = requests.post(url, headers=headers)

            if response.status_code == 204:
                return {
                    'status': 'success',
                    'message': 'Campaign sent'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to send campaign'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


class EmailCampaignTemplates:
    """Pre-built email templates for music releases."""

    @staticmethod
    def new_release_email(track_info: Dict) -> Dict[str, str]:
        """Generate new release email."""
        title = track_info.get('title', 'New Track')
        youtube_url = track_info.get('youtube_url', '#')
        spotify_url = track_info.get('spotify_url', '#')

        subject = f"🎵 New Release: {title}"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f4f4f4; padding: 30px; }}
        .cta-button {{ display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 10px 5px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 New LoFi Track</h1>
            <h2>{title}</h2>
        </div>

        <div class="content">
            <p>Hey there!</p>

            <p>I just dropped a new track and I think you're going to love it.</p>

            <p><strong>{title}</strong> is now available on all platforms:</p>

            <div style="text-align: center; margin: 30px 0;">
                <a href="{youtube_url}" class="cta-button">▶️ Watch on YouTube</a>
                <a href="{spotify_url}" class="cta-button">🎧 Stream on Spotify</a>
            </div>

            <p>Perfect for studying, working, or just vibing.</p>

            <p>Let me know what you think by leaving a comment!</p>

            <p>Peace,<br>LoFi AI</p>
        </div>

        <div class="footer">
            <p>You're receiving this because you subscribed to LoFi AI releases.</p>
            <p><a href="{{{{ unsubscribe }}}}">Unsubscribe</a></p>
        </div>
    </div>
</body>
</html>
"""

        return {
            'subject': subject,
            'preview_text': f"New {track_info.get('mood', 'chill')} beats just dropped 🔥",
            'html': html_content
        }

    @staticmethod
    def lead_magnet_email(download_link: str) -> Dict[str, str]:
        """Generate lead magnet delivery email (free sample pack)."""
        subject = "🎁 Your Free LoFi Sample Pack is Ready!"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
        .content {{ background: #f4f4f4; padding: 30px; }}
        .download-button {{ display: inline-block; background: #28a745; color: white; padding: 20px 40px; text-decoration: none; border-radius: 5px; font-size: 18px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎁 Your Free Sample Pack!</h1>
        </div>

        <div class="content">
            <p>Thanks for subscribing!</p>

            <p>Here's your free LoFi sample pack as promised:</p>

            <ul>
                <li>50+ Drum One-Shots</li>
                <li>20+ Melodic Loops</li>
                <li>10+ MIDI Files</li>
                <li>100% Royalty-Free</li>
            </ul>

            <div style="text-align: center; margin: 30px 0;">
                <a href="{download_link}" class="download-button">⬇️ Download Now</a>
            </div>

            <p>Enjoy creating amazing beats!</p>

            <p>Stay tuned for more free content and exclusive releases.</p>
        </div>
    </div>
</body>
</html>
"""

        return {
            'subject': subject,
            'preview_text': "Your free sample pack download link inside ✨",
            'html': html_content
        }


class EmailAutomation:
    """Complete email marketing automation."""

    def __init__(self, config: dict):
        """
        Initialize email automation.

        Args:
            config: Configuration with email service credentials
        """
        mailchimp_config = config.get('mailchimp', {})

        self.mailchimp = MailchimpManager(
            api_key=mailchimp_config.get('api_key'),
            server_prefix=mailchimp_config.get('server_prefix'),
            list_id=mailchimp_config.get('list_id')
        ) if mailchimp_config.get('api_key') else None

    def announce_new_release(self, track_info: Dict) -> Dict:
        """
        Announce new release to email list.

        Args:
            track_info: Track information

        Returns:
            Campaign result
        """
        if not self.mailchimp:
            return {
                'status': 'error',
                'message': 'Mailchimp not configured'
            }

        # Generate email content
        email = EmailCampaignTemplates.new_release_email(track_info)

        # Create campaign
        campaign = self.mailchimp.create_campaign(
            subject=email['subject'],
            preview_text=email['preview_text'],
            content=email['html']
        )

        if campaign['status'] == 'success':
            # Send immediately
            send_result = self.mailchimp.send_campaign(campaign['campaign_id'])
            return send_result
        else:
            return campaign

    def send_lead_magnet(self, email: str, download_link: str) -> Dict:
        """
        Send lead magnet (free sample pack) to new subscriber.

        Args:
            email: Subscriber email
            download_link: Download link for sample pack

        Returns:
            Send result
        """
        if not self.mailchimp:
            return {
                'status': 'error',
                'message': 'Mailchimp not configured'
            }

        # Add subscriber
        self.mailchimp.add_subscriber(email, tags=['lead_magnet'])

        # Generate and send welcome email with download
        email_content = EmailCampaignTemplates.lead_magnet_email(download_link)

        # Create and send campaign
        campaign = self.mailchimp.create_campaign(
            subject=email_content['subject'],
            preview_text=email_content['preview_text'],
            content=email_content['html']
        )

        if campaign['status'] == 'success':
            return self.mailchimp.send_campaign(campaign['campaign_id'])
        else:
            return campaign


if __name__ == '__main__':
    print("📧 Email Marketing Integration")
    print("=" * 60)

    # Demo
    print("\nTo enable email automation:")
    print("1. Sign up for Mailchimp")
    print("2. Get API key and list ID")
    print("3. Add to config.json:")
    print("""
    {
        "mailchimp": {
            "api_key": "your-api-key",
            "server_prefix": "us1",
            "list_id": "your-list-id"
        }
    }
    """)

    print("\nExample email templates:")
    demo_track = {
        'title': 'Midnight Study Session',
        'mood': 'chill',
        'youtube_url': 'https://youtube.com/...',
        'spotify_url': 'https://spotify.com/...'
    }

    email = EmailCampaignTemplates.new_release_email(demo_track)
    print(f"\nSubject: {email['subject']}")
    print(f"Preview: {email['preview_text']}")
