package com.scamcall.detection

import android.util.Log
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class ScamWebSocketClient(
    private val serverUrl: String,
    private val onScoreReceived: (Double, String) -> Unit,
    private val onStatusChanged: (String) -> Unit
) {
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    fun connect() {
        val request = Request.Builder().url(serverUrl).build()
        val listener = object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                onStatusChanged("Connected")
                Log.d("ScamWebSocket", "Connected")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d("ScamWebSocket", "Message received text: $text")
                try {
                    val json = JSONObject(text)
                    val riskScore = json.optDouble("risk_score", 0.0)
                    val riskLabel = json.optString("risk_label", "SAFE")
                    onScoreReceived(riskScore, riskLabel)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                onStatusChanged("Error: ${t.localizedMessage}")
                Log.e("ScamWebSocket", "Error: ${t.localizedMessage}")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                onStatusChanged("Disconnected")
                Log.d("ScamWebSocket", "Closed: $reason")
            }
        }
        
        webSocket = client.newWebSocket(request, listener)
    }

    fun sendAudioChunk(data: ByteArray) {
        webSocket?.send(data.toByteString())
    }

    fun disconnect() {
        webSocket?.close(1000, "App closed")
        webSocket = null
    }
}
