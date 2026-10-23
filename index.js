const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

app.post('/api/jarvis', async (req, res) => {
    try {
        const userPrompt = req.body.prompt || "Hello";
        
        const response = await axios.post(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`,
            {
                contents: [{
                    parts: [{
                        text: `You are JARVIS, a witty AI assistant. Keep response under 15 words: ${userPrompt}`
                    }]
                }]
            },
            { headers: { 'Content-Type': 'application/json' } }
        );

        const aiText = response.data.candidates[0].content.parts[0].text;
        res.json({ reply: aiText });
    } catch (error) {
        console.error("API Error:", error.message);
        res.status(500).json({ reply: "Server Error" });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`JARVIS Proxy running on port ${PORT}`));
