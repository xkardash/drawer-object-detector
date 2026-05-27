import 'package:flutter/material.dart';
import 'home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CekmeceApp());
}

class CekmeceApp extends StatelessWidget {
  const CekmeceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cekmece Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0E0E12),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF4D96FF),
          secondary: Color(0xFFFFD93D),
          surface: Color(0xFF1A1A20),
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
