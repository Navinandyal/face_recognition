from twilio.rest import Client

client = Client("ACc8982b79e38ee7ae94d7f0b1889910c2", "2b00fdd2079212ae4cd03e23d88c37b7")

msg = client.messages.create(
    body="🔥 Test WhatsApp working",
    from_='whatsapp:+14155238886',
    to='whatsapp:+918605083060'
)

print(msg.sid)