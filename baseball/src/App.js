
import './App.css';
import MainArea from './Component/MainArea';
import { initializeWebSocket } from './Component/WebServer';



initializeWebSocket()

function App() {
  return (
    <div className="App">
      <MainArea>

      </MainArea>
    </div>
  );
}

export default App;
