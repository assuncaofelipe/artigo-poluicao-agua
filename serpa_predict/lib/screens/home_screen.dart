import 'package:flutter/material.dart';

import '../services/data_store.dart';
import '../services/model_service.dart';
import 'manual_input_tab.dart';
import 'csv_tab.dart';
import 'map_tab.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  final ModelService _model = ModelService();
  final DataStore _store = DataStore();

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _model.init().then((_) => setState(() {}));
  }

  @override
  void dispose() {
    _tabController.dispose();
    _model.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Quality Monitoring'),
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF0F766E), Color(0xFF0EA5A4)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Manual Input'),
            Tab(text: 'CSV'),
            Tab(text: 'Map'),
          ],
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFF8FAFC), Color(0xFFE2F3F2)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: TabBarView(
          controller: _tabController,
          children: [
            ManualInputTab(model: _model, store: _store),
            CsvTab(model: _model, store: _store),
            MapTab(store: _store),
          ],
        ),
      ),
    );
  }
}
