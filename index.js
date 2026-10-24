const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

app.post('/api/jarvis', async (req, res) => {
    try {
        const userPrompt = req.body.prompt || "Hello";
        
        if (!GEMINI_API_KEY) {
            return res.status(500).json({ reply: "Error: GEMINI_API_KEY missing on Render!" });
        }

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
        const errorDetails = error.response ? JSON.stringify(error.response.data) : error.message;
        console.error("API Error Details:", errorDetails);
        res.status(500).json({ reply: "Gemini Error: " + errorDetails });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`JARVIS Proxy running on port ${PORT}`));
