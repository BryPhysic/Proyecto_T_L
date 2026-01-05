#!/bin/bash
# Monitor UMLS Indexing Progress

echo "🔍 Monitoreando progreso de indexación UMLS..."
echo "Presiona Ctrl+C para salir"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "📊 UMLS ChromaDB Indexing Progress"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if process is running
    if ps aux | grep -v grep | grep "build_umls_chromadb.py" > /dev/null; then
        echo "✅ Estado: CORRIENDO"
        
        # Get process info
        ps aux | grep -v grep | grep "build_umls_chromadb.py" | awk '{print "🖥️  CPU: " $3 "% | RAM: " $4 "% | Tiempo: " $10}'
        
        echo ""
        echo "📁 Verificando base de datos..."
        
        # Check database size
        if [ -d "Datasets/chromadb_umls" ]; then
            SIZE=$(du -sh Datasets/chromadb_umls 2>/dev/null | awk '{print $1}')
            echo "💾 Tamaño ChromaDB: $SIZE"
        else
            echo "⏳ Base de datos aún no creada..."
        fi
        
        echo ""
        echo "💡 Tip: El proceso tardará ~3-4 horas en total"
        echo "💡 Puedes cerrar esta ventana, seguirá corriendo en background"
        
    else
        echo "❌ Estado: NO CORRIENDO"
        echo ""
        
        if [ -d "Datasets/chromadb_umls" ]; then
            echo "✅ ¡Indexación completada!"
            SIZE=$(du -sh Datasets/chromadb_umls 2>/dev/null | awk '{print $1}')
            echo "💾 Tamaño final: $SIZE"
            break
        else
            echo "⚠️  El proceso no está corriendo y no hay base de datos"
        fi
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Actualizando en 10 segundos..."
    sleep 10
done
