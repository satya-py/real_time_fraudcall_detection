package com.scamcall.detection

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {

    private var isMonitoring by mutableStateOf(false)
    private var riskLabelState by mutableStateOf("SAFE")
    private var statusState by mutableStateOf("Status: Ready")

    private lateinit var audioStreamer: AudioStreamer
    private lateinit var webSocketClient: ScamWebSocketClient

    // Modern Android permissions handler in Activity/Compose
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions[Manifest.permission.RECORD_AUDIO] == true) {
            startMonitoring()
        } else {
            Toast.makeText(this, "Audio permission is required!", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize WebSocket Client (Update IP to your FastAPI server machine's IP)
        webSocketClient = ScamWebSocketClient(
            serverUrl = "ws://192.168.31.224:8000/ws/predict", // Updated to actual Local IP
            onScoreReceived = { score, label ->
                runOnUiThread {
                    riskLabelState = label
                }
            },
            onStatusChanged = { status ->
                runOnUiThread {
                    statusState = "Status: $status"
                }
            }
        )

        audioStreamer = AudioStreamer(webSocketClient)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ScamDetectionScreen()
                }
            }
        }
    }

    @Composable
    fun ScamDetectionScreen() {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Real-Time Scam Detector",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(24.dp))

            val riskColor = when (riskLabelState) {
                "SAFE" -> Color(0xFF4CAF50)
                "LOW RISK" -> Color(0xFFFFEB3B)
                "MODERATE RISK" -> Color(0xFFFF9800)
                "HIGH RISK" -> Color(0xFFF44336)
                "SCAM ALERT" -> Color(0xFFE91E63)
                else -> Color.Black
            }

            val displayText = if (riskLabelState == "SAFE") "✅ SAFE" else "🚨 $riskLabelState"

            Text(
                text = displayText,
                fontSize = 36.sp,
                fontWeight = FontWeight.Bold,
                color = riskColor
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = statusState,
                fontSize = 14.sp,
                color = Color.Gray
            )

            Spacer(modifier = Modifier.height(48.dp))

            Button(
                onClick = {
                    if (isMonitoring) {
                        stopMonitoring()
                    } else {
                        if (checkPermissions()) {
                            startMonitoring()
                        } else {
                            requestPermissionLauncher.launch(
                                arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.INTERNET)
                            )
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(60.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isMonitoring) Color(0xFFE53935) else Color(0xFF4CAF50)
                )
            ) {
                Text(
                    text = if (isMonitoring) "Stop Monitoring" else "Start Monitoring Call",
                    fontSize = 18.sp,
                    color = Color.White
                )
            }
        }
    }

    private fun checkPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startMonitoring() {
        isMonitoring = true
        statusState = "Status: Connecting..."
        
        webSocketClient.connect()
        // Wait briefly for connection before streaming
        Handler(Looper.getMainLooper()).postDelayed({
            audioStreamer.startStreaming()
        }, 1000)
    }

    private fun stopMonitoring() {
        isMonitoring = false
        statusState = "Status: Disconnected"
        
        audioStreamer.stopStreaming()
        webSocketClient.disconnect()
        riskLabelState = "SAFE"
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isMonitoring) {
            stopMonitoring()
        }
    }
}
